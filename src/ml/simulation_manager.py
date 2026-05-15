import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Optional
import tensorflow as tf
import os
import json
import config
from backend.twilio_service import send_alert
from src.data.generator_v2.engine import VitalsGeneratorV2
from src.data.generator_v2.clinical_logic import ClinicalStateEngine

from src.ml.model_registry import get_predictor

class SimulationManager:
    def __init__(self, model_path: str):
        self.active_sessions = {}
        self.CLASS_NAMES = [
            'Stable', 'Monitor', 'Heart_Attack', 'Arrhythmia', 'Heart_Failure',
            'Hypoglycemia', 'Hyperglycemia_DKA', 'Respiratory_Distress', 'Sepsis',
            'Stroke', 'Shock', 'Hypertensive_Crisis', 'Fall_Unconscious'
        ]

    def initialize_patient(self, patient_id: str, name: str, age: int, gender: str, conditions: List[str], activity_level: str = 'moderate', contacts: List[Dict] = None):
        age = int(age)
        generator = VitalsGeneratorV2(age=age, gender=gender, conditions=conditions, duration=600)
        episode_data, _ = generator.generate_episode()
        # Pre-populate the buffer with the full 600 seconds of data
        buffer = deque(episode_data.tolist(), maxlen=600)
        self.active_sessions[patient_id] = {
            'patient_id': patient_id, 'name': name, 'age': age, 'gender': gender,
            'conditions': conditions, 'activity_level': activity_level, 'contacts': contacts or [],
            'mode': 'stable', 'simulation_active': False, 'buffer': buffer, 'generator': generator,
            'emergency_scenario': None, 'last_vitals': np.array(buffer[-1]), 'alert_stage': 'green',
            'consecutive_ticks': 0, 'rationale': "Vitals within normal variance.",
            'window_tick': 0, 'total_ticks': 0
        }

    def trigger_emergency(self, patient_id: str, scenario_idx: int):
        session = self.active_sessions.get(patient_id)
        if not session: raise Exception(f"Session {patient_id} not found.")
        
        # Reset any active simulation artifacts before starting new one
        if session.get('simulation_active'):
            print(f"DEBUG: Resetting active simulation for patient {patient_id} before override")
            session.pop('emergency_generator', None)
            session.pop('stable_buffer_snapshot', None)
        
        print(f"DEBUG: Triggering emergency scenario {scenario_idx} for patient {patient_id}")
        
        # Initialize the V2 Generator for the specific emergency
        generator = VitalsGeneratorV2(
            age=session['age'], 
            gender=session['gender'], 
            initial_state=scenario_idx,
            conditions=session['conditions'],
            duration=600
        )
        
        # Pre-generate the emergency episode data
        emergency_data, _ = generator.generate_episode()
        
        # Update session state with emergency artifacts
        session.update({
            'simulation_active': True,
            'mode': 'transitioning',
            'emergency_scenario': scenario_idx,
            'transition_tick': 1, # Start at 1 to ensure immediate alpha shift
            'window_tick': 0,
            'emergency_generator': emergency_data,
            'stable_buffer_snapshot': deque(list(session['buffer']), maxlen=600)
        })
        print(f"DEBUG: Emergency state initialized successfully for patient {patient_id}")

    def get_next_tick(self, patient_id: str, request_history: bool = False) -> Dict[str, Any]:
        session = self.active_sessions.get(patient_id)
        if not session: return {"error": f"Session not found"}
        
        label, confidence, status = "Stable", 1.0, "stable" # Defaults
        
        try:
            # 1. GENERATE THE TICK DATA
            if session['mode'] == 'stable':
                if not session.get('current_episode_buffer'):
                    episode_data, _ = session['generator'].generate_episode()
                    session['current_episode_buffer'] = deque(episode_data.tolist())
                tick = np.array(session['current_episode_buffer'].popleft())
            
            elif session['mode'] == 'transitioning':
                # Blend stable buffer with emergency data
                if session.get('stable_buffer_snapshot') and len(session['stable_buffer_snapshot']) > 0:
                    stable = np.array(session['stable_buffer_snapshot'].popleft())
                else:
                    stable = np.array(session['last_vitals'])
                
                emergency = np.array(session['emergency_generator'][session['transition_tick']])
                alpha = session['transition_tick'] / 30.0
                tick = (1.0 - alpha) * stable + alpha * emergency
                
                session['transition_tick'] += 1
                if session['transition_tick'] >= 30:
                    session['mode'] = 'emergency'
                    session['window_tick'] = 0
            
            else: # Emergency mode: Continuous data feed
                gen_data = session['emergency_generator']
                tick = np.array(gen_data[session['window_tick']])
                session['window_tick'] = (session['window_tick'] + 1) % len(gen_data)

            # 2. UPDATE BUFFERS
            session['buffer'].append(tick.tolist())
            session['last_vitals'] = tick
            session['total_ticks'] += 1

            # 3. RUN MODEL INFERENCE (Robust Buffer Padding)
            raw_buffer = list(session['buffer'])
            input_array = np.array(raw_buffer, dtype=np.float32)
            
            # Robust Padding: Always pass (600, 7) to model
            if input_array.shape[0] < 600:
                padded_input = np.zeros((600, 7), dtype=np.float32)
                padded_input[-input_array.shape[0]:] = input_array
                input_data = padded_input.reshape(1, 600, 7)
            else:
                input_data = input_array[-600:].reshape(1, 600, 7)
            
            _, confidence, label, probs = get_predictor().predict(input_data.reshape(600, 7))
            status = self._determine_status(self.CLASS_NAMES.index(label), confidence)
            self._update_alert_stage(session, confidence, label)
            
            # Use the engine's current profile for baselines
            baselines = session['generator'].profile.baselines
            session['rationale'] = self._generate_rationale(tick, label, confidence, baselines)
        
        except Exception as e:
            print(f"ERROR: Simulation tick failure for patient {patient_id}: {str(e)}")
            import traceback; traceback.print_exc()
            tick = np.array(session['last_vitals'])

        response = {
            "ts": np.datetime64('now').astype(str), 
            "vitals": dict(zip(['heart_rate','systolic_bp','diastolic_bp','spo2','temperature','resp_rate','blood_glucose'], tick.tolist())), 
            "prediction": {"label": label, "confidence": confidence, "rationale": session.get('rationale', "Analyzing...")}, 
            "status": status, 
            "alertStage": session.get('alert_stage', 'green'), 
            "simulation_active": session.get('simulation_active', False),
            "emergency_scenario": session.get('emergency_scenario'),
            "mode": session['mode'],
            "emergency_contacts": session['contacts']
        }
        
        if request_history:
            response["history"] = [dict(zip(['heart_rate','systolic_bp','diastolic_bp','spo2','temperature','resp_rate','blood_glucose'], v)) for v in list(session['buffer'])]
            
        return self._sanitize_floats(response)

    def abort_simulation(self, patient_id: str):
        session = self.active_sessions.get(patient_id)
        if not session: raise Exception(f"Session {patient_id} not found.")
        
        session.update({
            'mode': 'stable', 
            'simulation_active': False, 
            'emergency_scenario': None,
            'transition_tick': 0,
            'window_tick': 0
        })
        session.pop('emergency_generator', None)
        session.pop('stable_buffer_snapshot', None)
        print(f"DEBUG: Emergency simulation aborted for patient {patient_id}")

    def _determine_status(self, pred_idx: int, confidence: float) -> str:
        # 0: Stable, 1: Monitor, 2-9: Emergency, 10+: Critical
        if pred_idx == 0: return "stable"
        if pred_idx == 1: return "monitor"
        if pred_idx < 10: return "emergency"
        return "critical"

    def _update_alert_stage(self, session: Dict, confidence: float, label: str):
        if confidence > 0.40: session['consecutive_ticks'] += 1
        else: session['consecutive_ticks'] = 0
        new_stage = session['alert_stage']
        if confidence >= 0.95: new_stage = 'red'
        elif session['alert_stage'] == 'green' and session['consecutive_ticks'] >= 3 and confidence > 0.40: new_stage = 'yellow'
        elif session['alert_stage'] == 'yellow' and session['consecutive_ticks'] >= 5 and confidence > 0.72: new_stage = 'orange'
        elif session['alert_stage'] == 'orange' and session['consecutive_ticks'] >= 3 and confidence >= 0.85: new_stage = 'red'
        if new_stage != session['alert_stage']:
            session['alert_stage'] = new_stage
            if new_stage in ['orange', 'red']: self._dispatch_sms(session, new_stage, label)

    def _dispatch_sms(self, session: Dict, stage: str, label: str):
        for contact in session.get('contacts', []):
            if contact.get('phone'): send_alert(session.get('name', 'Patient'), f"[{stage.upper()}] {label.replace('_', ' ')}", contact['phone'])

    def _sanitize_floats(self, obj):
        if isinstance(obj, float): return 0.0 if not np.isfinite(obj) else obj
        if isinstance(obj, dict): return {k: self._sanitize_floats(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._sanitize_floats(v) for v in obj]
        return obj

    def _generate_rationale(self, vitals: np.ndarray, label: str, conf: float, baselines: Dict) -> str:
        label_human = label.replace('_', ' ')
        conf_prefix = ""
        if conf < 0.60: conf_prefix = "Low Confidence: "
        elif conf < 0.85: conf_prefix = "Probable: "
        
        return f"{conf_prefix}{label_human} ({conf*100:.1f}%)."

sim_manager = None
def get_sim_manager():
    global sim_manager
    if sim_manager is None: sim_manager = SimulationManager(config.MODEL_V1_2_H5_PATH)
    return sim_manager

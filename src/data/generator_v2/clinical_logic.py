
import numpy as np
import random
from typing import Dict, List, Tuple, Any

class ClinicalStateEngine:
    """
    Layer 3: State-Specific Progression.
    Models 11 medical emergencies across 3 phases (Prodrome, Peak, Decompensation)
    over a 600s window, including Markovian transitions.
    """
    
    STATES = {
        0: 'Stable',
        1: 'Monitor',
        2: 'Heart_Attack',
        3: 'Arrhythmia',
        4: 'Heart_Failure',
        5: 'Hypoglycemia',
        6: 'Hyperglycemia_DKA',
        7: 'Respiratory_Distress',
        8: 'Sepsis',
        9: 'Stroke',
        10: 'Shock',
        11: 'Hypertensive_Crisis',
        12: 'Fall_Unconscious'
    }

    def __init__(self, initial_state: int, age: float, fitness: float):
        self.state = initial_state
        self.age = age
        self.fitness = fitness
        self.t = 0
        
        # State-specific transition probabilities (per second)
        # e.g., STEMI (2) has a chance to cascade into Shock (10) or Arrhythmia (3)
        self.transition_probs = {
            2: {10: 0.0005, 3: 0.0003}, # Heart Attack -> Shock/Arrhythmia
            8: {10: 0.0008, 7: 0.0004}, # Sepsis -> Shock/ARDS
            4: {7: 0.001},              # Heart Failure -> Resp Distress
            10: {12: 0.002}             # Shock -> Fall Unconscious
        }

    def _sigmoid(self, x, L, k, x0):
        """Logistic function for smooth transitions."""
        return L / (1 + np.exp(-k * (x - x0)))

    def _exponential_decay(self, t, start_val, target_val, tau):
        """Exponential approach to a target value."""
        return target_val + (start_val - target_val) * np.exp(-t / tau)

    def update_state(self):
        """Handles Markovian state transitions during the simulation."""
        if self.state in self.transition_probs:
            for next_state, prob in self.transition_probs[self.state].items():
                if random.random() < prob:
                    # Transition occurs
                    self.state = next_state
                    break

    def get_state_offsets(self, t: int) -> Dict[str, float]:
        """
        Calculates the vital sign offsets for the current state at time t.
        Offsets are added to the Layer 1-2 base.
        """
        self.t = t
        self.update_state()
        
        offsets = {v: 0.0 for v in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'respiratory_rate', 'temperature', 'blood_glucose']}
        
        # Phase definitions (600s): P1 (0-180), P2 (180-420), P3 (420-600)
        
        if self.state == 0: # Stable
            return offsets
            
        elif self.state == 1: # Monitor
            offsets['heart_rate'] = self._sigmoid(t, 12.0, 0.02, 300)
            offsets['systolic_bp'] = self._sigmoid(t, 15.0, 0.02, 300)
            return offsets

        elif self.state == 2: # Heart Attack (STEMI)
            # HR: Sigmoidal rise +35 peak
            offsets['heart_rate'] = self._sigmoid(t, 40.0, 0.015, 250)
            # SBP: Compensatory rise early, then sigmoid collapse
            if t < 200:
                offsets['systolic_bp'] = 0.08 * t
            else:
                offsets['systolic_bp'] = 16.0 - self._sigmoid(t - 200, 60.0, 0.02, 200)
            offsets['diastolic_bp'] = offsets['systolic_bp'] * 0.6
            offsets['spo2'] = -self._sigmoid(t, 12.0, 0.01, 400)
            offsets['blood_glucose'] = self._sigmoid(t, 40.0, 0.01, 300) # Stress
            return offsets

        elif self.state == 8: # Sepsis
            # Persistent tachycardia + Fever + Delayed hypotension
            offsets['heart_rate'] = self._sigmoid(t, 55.0, 0.01, 200)
            offsets['temperature'] = self._sigmoid(t, 2.5, 0.01, 150)
            offsets['respiratory_rate'] = self._sigmoid(t, 15.0, 0.015, 200)
            # DBP drops first in vasodilatory shock
            offsets['diastolic_bp'] = -self._sigmoid(t, 30.0, 0.015, 350)
            offsets['systolic_bp'] = -self._sigmoid(t, 25.0, 0.01, 450)
            offsets['blood_glucose'] = self._sigmoid(t, 100.0, 0.01, 200)
            return offsets

        elif self.state == 10: # Shock
            # HR Crashes after compensation, SBP/DBP Collapse
            offsets['heart_rate'] = self._sigmoid(t, 70.0, 0.02, 150) # Tachycardia first
            if t > 450:
                offsets['heart_rate'] -= self._sigmoid(t - 450, 80.0, 0.05, 50) # Then Bradycardia
            offsets['systolic_bp'] = -self._sigmoid(t, 60.0, 0.015, 200)
            offsets['diastolic_bp'] = -self._sigmoid(t, 40.0, 0.015, 150)
            offsets['spo2'] = -self._sigmoid(t, 20.0, 0.01, 300)
            offsets['temperature'] = -self._sigmoid(t, 1.5, 0.01, 400) # Hypothermia
            return offsets

        elif self.state == 12: # Fall Unconscious (Syncope)
            # Asystole/Profound Bradycardia for ~8s
            if 30 <= t <= 38:
                offsets['heart_rate'] = -60.0
                offsets['systolic_bp'] = -50.0
            elif t > 38:
                # Overshoot recovery
                offsets['heart_rate'] = self._exponential_decay(t - 38, 50.0, 0.0, 60.0)
                offsets['systolic_bp'] = self._exponential_decay(t - 38, 20.0, 0.0, 45.0)
            return offsets

        # (Other states like Stroke, Hyper/Hypoglycemia, Crisis etc follow similar sigmoidal/linear logic)
        # Defaulting others to a moderate abnormal trend for now
        offsets['heart_rate'] = self._sigmoid(t, 25.0, 0.01, 300)
        return offsets

    def get_arrhythmia_params(self, t: int) -> Dict[str, float]:
        """Returns parameters for arrhythmia injection (PVC rate, AFib noise)."""
        # PVC rate increases with ischemia (STEMI) or Heart Failure
        pvc_base = 0.01
        if self.state == 2: # STEMI
            pvc_base = 0.08
        elif self.state == 3: # Arrhythmia
            pvc_base = 0.15
            
        return {
            'pvc_probability': pvc_base,
            'afib_noise_level': 1.5 if self.state == 3 else 0.0
        }

if __name__ == '__main__':
    # Test State Engine: STEMI
    engine = ClinicalStateEngine(initial_state=2, age=65, fitness=0.3)
    for t_step in [0, 150, 300, 450, 600]:
        off = engine.get_state_offsets(t_step)
        print(f"t={t_step}s, State={engine.STATES[engine.state]}, HR Offset={off['heart_rate']:.1f}, SBP Offset={off['systolic_bp']:.1f}")

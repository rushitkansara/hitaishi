
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
import json
import os

from .oscillators import PhysiologicalOscillator
from .patient_profile import PatientProfile
from .coupling import AutonomicCoupling
from .clinical_logic import ClinicalStateEngine
from .artifacts import TelemetryArtifacts

class VitalsGeneratorV2:
    """
    Hierarchical Stochastic Vital Signs Generator (v2.0).
    Orchestrates 5 layers of physiology:
    L1: Oscillators (fBM, RSA, Mayer)
    L2: Baseline + Demographics
    L3: State Progression (Emergency Cascades)
    L4: Autonomic Coupling (Lags)
    L5: Measurement Realism (Artifacts, Loss)
    """
    
    def __init__(self, age: float, gender: str, initial_state: int = 0, 
                 fitness: float = 0.5, conditions: Optional[List[str]] = None, 
                 duration: int = 600, l_matrix: Optional[np.ndarray] = None):
        self.duration = duration
        
        # Initialize sub-modules
        self.profile = PatientProfile(age, gender, fitness, conditions)
        self.oscillator = PhysiologicalOscillator(n=duration, h=0.75, l_matrix=l_matrix)
        self.coupling = AutonomicCoupling(self.profile.baselines, age)
        self.clinical = ClinicalStateEngine(initial_state, age, fitness)
        self.artifacts = TelemetryArtifacts(age)
        
        self.limits = self.profile.get_limits()

    def generate_episode(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs the 600s simulation loop.
        Returns:
            A NumPy array of shape (600, 7) and a metadata dictionary.
        """
        vitals_history = []
        state_log = []
        
        # Precompute L1-L2 Base noise and cycles (Non-interactive)
        fgn_hr = self.oscillator.generate_fgn(sigma=0.8)
        fgn_bp = self.oscillator.generate_fgn(sigma=1.2)
        mayer = self.oscillator.mayer_wave()
        
        # FIX: Move respiratory cycle OUTSIDE the loop
        rr_cycle = self.oscillator.respiratory_cycle(self.profile.baselines['respiratory_rate'], sigma=0.4)
        hr_rsa_full = self.oscillator.rsa_term(self.profile.baselines['respiratory_rate'] + rr_cycle, amplitude=3.5)
        
        # Loop for 600 seconds
        for t in range(self.duration):
            # --- LAYER 1: Oscillators (Wiggles) ---
            current_rr = self.profile.baselines['respiratory_rate'] + rr_cycle[t]
            hr_rsa = hr_rsa_full[t]
            
            # --- LAYER 3: State Progression (Trends) ---
            offsets = self.clinical.get_state_offsets(t)
            arr_params = self.clinical.get_arrhythmia_params(t)
            state_log.append((t, self.clinical.state))
            
            # --- LAYER 4: Autonomic Coupling (Interactions) ---
            baro_adj = self.coupling.get_baroreflex_adjustment()
            chemo_adj = self.coupling.get_chemoreflex_adjustment()
            metab_adj = self.coupling.get_metabolic_coupling(self.profile.baselines['heart_rate'] + offsets['heart_rate'])
            mechno_adj = self.coupling.get_mechanical_ventilation_effect()
            
            # --- SUMMATION ---
            current_vitals = {
                'heart_rate': (self.profile.baselines['heart_rate'] + 
                               offsets['heart_rate'] + 
                               hr_rsa + fgn_hr[t] + 
                               baro_adj + chemo_adj['heart_rate']),
                               
                'systolic_bp': (self.profile.baselines['systolic_bp'] + 
                                offsets['systolic_bp'] + 
                                mayer[t] + fgn_bp[t] + 
                                mechno_adj),
                                
                'diastolic_bp': (self.profile.baselines['diastolic_bp'] + 
                                 offsets['diastolic_bp'] + 
                                 (mayer[t] * 0.6) + (fgn_bp[t] * 0.4)),
                                 
                'spo2': (self.profile.baselines['spo2'] + 
                         offsets['spo2'] + 
                         (rr_cycle[t] * 0.2)), # Small Resp-SpO2 coupling
                         
                'respiratory_rate': (current_rr + 
                                     offsets['respiratory_rate'] + 
                                     chemo_adj['respiratory_rate']),
                                     
                'temperature': (self.profile.baselines['temperature'] + 
                                offsets['temperature'] + 
                                metab_adj['temperature']),
                                
                'blood_glucose': (self.profile.baselines['blood_glucose'] + 
                                  offsets['blood_glucose'] + 
                                  metab_adj['blood_glucose'])
            }
            
            # --- LAYER 5: Measurement Artifacts + Limits ---
            v_noisy = self.artifacts.apply_noise(current_vitals)
            v_lost = self.artifacts.inject_data_loss(v_noisy)
            v_final = self.artifacts.enforce_physiological_limits(v_lost, self.limits)
            
            # Update buffers for next timestep coupling
            self.coupling.update_buffers(v_final)
            
            # Store in result array (Correct Feature Mapping)
            # 0:hr, 1:sbp, 2:dbp, 3:spo2, 4:temp, 5:rr, 6:glucose
            vitals_history.append([
                v_final['heart_rate'], v_final['systolic_bp'], v_final['diastolic_bp'],
                v_final['spo2'], v_final['temperature'], v_final['respiratory_rate'],
                v_final['blood_glucose']
            ])
            
        metadata = {
            'patient_age': self.profile.age,
            'gender': self.profile.gender,
            'initial_state': self.clinical.STATES[state_log[0][1]],
            'final_state': self.clinical.STATES[self.clinical.state],
            'transitions': state_log
        }
        
        return np.array(vitals_history), metadata

    def validate_physiological_sanity(self, data: np.ndarray) -> Dict[str, Any]:
        """Runs online sanity checks to verify realism."""
        hr = data[:, 0]
        sbp = data[:, 1]
        
        # Pearson HR-SBP Correlation (exclude NaNs)
        valid_idx = ~np.isnan(hr) & ~np.isnan(sbp)
        correlation = np.corrcoef(hr[valid_idx], sbp[valid_idx])[0, 1]
        
        # Autocorrelation (Smoothness)
        hr_valid = hr[~np.isnan(hr)]
        from statsmodels.tsa.stattools import acf
        acf_1 = acf(hr_valid, nlags=1, fft=False)[1]
        
        return {
            'hr_sbp_correlation': correlation,
            'hr_autocorrelation_lag1': acf_1,
            'realism_passed': (correlation > 0.1 or correlation < -0.1) and acf_1 > 0.95
        }

if __name__ == '__main__':
    # Full Demo Run: 80yo Hypertension -> Sepsis
    print("--- Running Generation V2 Test ---")
    gen = VitalsGeneratorV2(age=80, gender='female', initial_state=8, conditions=['Hypertension'])
    data, meta = gen.generate_episode()
    
    sanity = gen.validate_physiological_sanity(data)
    print(f"Episode Generated: {meta['initial_state']} -> {meta['final_state']}")
    print(f"Sanity: Correlation={sanity['hr_sbp_correlation']:.4f}, Smoothness={sanity['hr_autocorrelation_lag1']:.4f}")
    print(f"Realism Pass: {sanity['realism_passed']}")
    
    # Save a small sample for inspection
    np.save('v2_demo_episode.npy', data)
    print("Demo episode saved to v2_demo_episode.npy")

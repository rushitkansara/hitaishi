
import numpy as np
from collections import deque
from typing import Dict, List, Any

class AutonomicCoupling:
    """
    Layer 4: Autonomic Coupling (Cross-Vital Interactions).
    Handles physiological feedback loops with realistic lags using buffers.
    """
    
    def __init__(self, baselines: Dict[str, float], age: float):
        self.baselines = baselines
        self.age = age
        
        # Buffers for lag handling (max lag 180s for central chemo)
        self.buffer_size = 200
        self.buffers = {
            'heart_rate': deque([baselines['heart_rate']] * self.buffer_size, maxlen=self.buffer_size),
            'systolic_bp': deque([baselines['systolic_bp']] * self.buffer_size, maxlen=self.buffer_size),
            'spo2': deque([baselines['spo2']] * self.buffer_size, maxlen=self.buffer_size),
            'respiratory_rate': deque([baselines['respiratory_rate']] * self.buffer_size, maxlen=self.buffer_size),
            'glucose_integral': 0.0
        }
        
        # Age-based reflex sensitivity (Elderly have blunted reflexes)
        self.sensitivity = max(0.3, 1.0 - (self.age / 120.0))

    def update_buffers(self, current_vitals: Dict[str, float]):
        """Steps the buffers forward with the latest values."""
        for key in ['heart_rate', 'systolic_bp', 'spo2', 'respiratory_rate']:
            if key in current_vitals:
                self.buffers[key].append(current_vitals[key])

    def get_baroreflex_adjustment(self) -> float:
        """
        Baroreflex: HR inversely tracks SBP changes with a ~10-beat lag.
        Lag: ~8-10 seconds depending on HR.
        """
        # We use a 10-second lag as a proxy for 10 beats at 60bpm
        lag_idx = -10
        past_sbp = self.buffers['systolic_bp'][lag_idx]
        sbp_diff = past_sbp - self.baselines['systolic_bp']
        
        # Coefficient: -0.18 (Inverse relationship)
        adjustment = -0.18 * sbp_diff * self.sensitivity
        return adjustment

    def get_chemoreflex_adjustment(self) -> Dict[str, float]:
        """
        Peripheral Chemoreflex: SpO2 drops trigger RR and HR increases.
        Lag: ~20 seconds for peripheral (carotid body).
        """
        lag_idx = -20
        past_spo2 = self.buffers['spo2'][lag_idx]
        spo2_deficit = max(0.0, self.baselines['spo2'] - past_spo2)
        
        # Only triggers significantly below 92%
        trigger = 1.0 if past_spo2 < 92.0 else 0.2
        
        adjustments = {
            'respiratory_rate': 1.8 * spo2_deficit * trigger * self.sensitivity,
            'heart_rate': 0.8 * spo2_deficit * trigger * self.sensitivity
        }
        return adjustments

    def get_metabolic_coupling(self, current_hr: float) -> Dict[str, float]:
        """
        Metabolic: Sustained tachycardia increases Glucose (stress) and Temp (heat).
        """
        hr_excess = max(0.0, current_hr - self.baselines['heart_rate'] - 10.0)
        
        # Stress hyperglycemia (Catecholamine release)
        # Slow accumulation: +0.02 mg/dL per second of excess HR
        self.buffers['glucose_integral'] += 0.02 * hr_excess
        
        # Thermal inertia (Slow rise)
        temp_rise = 0.0005 * self.buffers['glucose_integral']
        
        return {
            'blood_glucose': 0.05 * hr_excess, # Instant stress spike
            'temperature': temp_rise
        }

    def get_mechanical_ventilation_effect(self) -> float:
        """
        Mechano-ventilatory: SBP drops slightly during deep inhalation (Preload reduction).
        Lag: 3 seconds after RR spike.
        """
        lag_idx = -3
        past_rr = self.buffers['respiratory_rate'][lag_idx]
        rr_diff = past_rr - self.baselines['respiratory_rate']
        
        # SBP drops ~0.9 mmHg per extra breath/min
        return -0.9 * rr_diff

if __name__ == '__main__':
    # Test Coupling
    base = {'heart_rate': 72, 'systolic_bp': 120, 'spo2': 98, 'respiratory_rate': 16}
    coupling = AutonomicCoupling(base, age=30)
    
    # Simulate a sudden BP spike
    test_vitals = base.copy()
    test_vitals['systolic_bp'] = 160
    
    # Fill buffer with high BP
    for _ in range(20):
        coupling.update_buffers(test_vitals)
        
    adj_hr = coupling.get_baroreflex_adjustment()
    print(f"HR adjustment after SBP spike: {adj_hr:.2f} BPM (Expected: Negative)")
    
    # Simulate low SpO2
    test_vitals['spo2'] = 85
    for _ in range(30):
        coupling.update_buffers(test_vitals)
        
    adj_chemo = coupling.get_chemoreflex_adjustment()
    print(f"RR adjustment after SpO2 drop: {adj_chemo['respiratory_rate']:.2f} Br/min")

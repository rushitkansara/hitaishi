
import numpy as np
import random
from typing import Dict, List, Any, Optional

class TelemetryArtifacts:
    """
    Layer 5: Measurement Realism.
    Simulates sensor noise, motion artifacts, and intermittent data loss (NaNs).
    Also enforces physiological clipping with "bounce-back" logic.
    """
    
    def __init__(self, age: float):
        self.age = age
        # Higher data loss rate for elderly due to poor perfusion/stiff arteries
        self.base_loss_rate = 0.12 + (0.05 * (age / 80.0))

    def apply_noise(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """Adds Gaussian and impulsive noise to simulate sensor jitter."""
        noisy = vitals.copy()
        
        # 1. Heart Rate Noise (ECG jitter)
        noisy['heart_rate'] += np.random.normal(0, 0.8)
        if random.random() < 0.02: # 2% chance of a motion spike
            noisy['heart_rate'] += random.choice([-5, 5])
            
        # 2. BP Noise (Cuff/Line variability)
        noisy['systolic_bp'] += np.random.normal(0, 1.5)
        noisy['diastolic_bp'] += np.random.normal(0, 1.0)
        
        # 3. SpO2 Noise (Finger motion)
        noisy['spo2'] += np.random.normal(0, 0.2)
        if random.random() < 0.03: # Motion artifact drops SpO2 temporarily
            noisy['spo2'] -= random.uniform(1.0, 3.0)
            
        # 4. RR Noise
        noisy['respiratory_rate'] += np.random.normal(0, 0.3)
        
        return noisy

    def inject_data_loss(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """Simulates intermittent sensor failure without creating NaNs."""
        processed = vitals.copy()
        
        # Reduced loss rates for demonstration
        if random.random() < (self.base_loss_rate * 0.2): 
            processed['spo2'] = 0.0 # Use 0 instead of NaN to avoid graph issues
        if random.random() < 0.01: 
            processed['heart_rate'] = 0.0 # Use 0 instead of NaN to avoid graph issues
            
        return processed

    def enforce_physiological_limits(self, vitals: Dict[str, float], limits: Dict[str, tuple]) -> Dict[str, float]:
        """Ensures vitals stay within possible bounds with soft 'bounce-back'."""
        clipped = vitals.copy()
        for key, (min_val, max_val) in limits.items():
            val = clipped[key]
            if np.isnan(val): continue
            
            if val > max_val:
                clipped[key] = max_val - abs(np.random.normal(0, 1.0))
            elif val < min_val:
                clipped[key] = min_val + abs(np.random.normal(0, 1.0))
                
        return clipped

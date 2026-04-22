
import numpy as np
from typing import Dict, List, Any, Optional

class PatientProfile:
    """
    Layer 2: Baseline + Demographics.
    Calculates physiological baselines and constraints (HR Max, Pulse Pressure) 
    based on age, gender, fitness, and preexisting conditions.
    """
    
    def __init__(self, age: float, gender: str, fitness_level: float = 0.5, 
                 conditions: Optional[List[str]] = None):
        """
        Parameters:
            age: Patient age in years.
            gender: 'male', 'female', or 'other'.
            fitness_level: 0.0 (sedentary) to 1.0 (athlete).
            conditions: List of strings (e.g., ['Hypertension', 'COPD']).
        """
        self.age = age
        self.gender = gender
        self.fitness_level = fitness_level
        self.conditions = conditions if conditions else []
        self.baselines = self._calculate_baselines()
        
    def _calculate_baselines(self) -> Dict[str, float]:
        """
        Computes the starting 7 vitals based on demographics and comorbidities.
        Reference: Tanaka, H., Monahan, K.D., & Seals, D.R. (2001).
        """
        # --- 1. Heart Rate (BPM) ---
        # Tanaka Formula for HR Max: 208 - 0.7 * age
        self.hr_max = 208 - 0.7 * self.age
        # Resting HR: Standard 72, shifted by fitness
        hr_base = 72.0 - (15.0 * self.fitness_level) # Athletes have lower HR
        
        # --- 2. Blood Pressure (mmHg) ---
        # Pulse Pressure widens with age (Arterial stiffness)
        # SBP usually rises, DBP often drops or stays stable in elderly
        pp_base = 40.0 + (0.4 * max(0, self.age - 30)) # PP widens after 30
        sbp_base = 110.0 + (0.6 * self.age) # SBP rises linearly with age
        
        # --- 3. Other Vitals ---
        resp_base = 14.0 + (0.05 * self.age) # RR rises slightly with age
        spo2_base = 99.0 - (0.02 * self.age) # SpO2 drops slightly with age
        temp_base = 36.8 # Celsius
        glucose_base = 85.0 + (0.2 * self.age) # Glucose rises with age
        
        # --- 4. Gender Adjustments ---
        if self.gender == 'female':
            hr_base += 6.0
            sbp_base -= 5.0
            
        # --- 5. Preexisting Condition Shifts ---
        if 'Hypertension' in self.conditions:
            sbp_base += 25.0
            pp_base += 10.0
        if 'COPD' in self.conditions:
            spo2_base -= 4.5
            resp_base += 3.5
            hr_base += 5.0
        if 'Diabetes' in self.conditions:
            glucose_base += 45.0
        if 'Athletic' in self.conditions or self.fitness_level > 0.8:
            hr_base -= 5.0
            sbp_base -= 8.0
        if 'Obesity' in self.conditions:
            sbp_base += 12.0
            hr_base += 6.0
        if 'Heart Failure' in self.conditions:
            hr_base += 10.0
            sbp_base -= 10.0
            spo2_base -= 2.0
            
        # Ensure DBP calculation from SBP and Pulse Pressure
        dbp_base = sbp_base - pp_base
        
        # Clinical clipping for baseline sanity
        return {
            'heart_rate': float(np.clip(hr_base, 40, 100)),
            'systolic_bp': float(np.clip(sbp_base, 90, 180)),
            'diastolic_bp': float(np.clip(dbp_base, 50, 110)),
            'spo2': float(np.clip(spo2_base, 88, 100)),
            'temperature': temp_base,
            'respiratory_rate': float(np.clip(resp_base, 10, 24)),
            'blood_glucose': float(np.clip(glucose_base, 70, 180))
        }

    def get_limits(self) -> Dict[str, tuple]:
        """Returns physiological hard-limits for this specific patient."""
        return {
            'heart_rate': (30, self.hr_max),
            'systolic_bp': (60, 260),
            'diastolic_bp': (40, 160),
            'spo2': (65, 100),
            'temperature': (34.0, 42.0),
            'respiratory_rate': (4, 60),
            'blood_glucose': (20, 800)
        }

if __name__ == '__main__':
    # Test: Elderly Hypertensive
    p1 = PatientProfile(age=82, gender='female', conditions=['Hypertension'])
    print(f"Elderly Hypertensive: {p1.baselines}")
    print(f"Pulse Pressure: {p1.baselines['systolic_bp'] - p1.baselines['diastolic_bp']:.1f}")
    
    # Test: Young Athlete
    p2 = PatientProfile(age=22, gender='male', fitness_level=0.9)
    print(f"\nYoung Athlete: {p2.baselines}")
    print(f"HR Max: {p2.hr_max:.1f}")

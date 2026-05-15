import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import new VitalsGeneratorV2
from src.data.generator_v2.engine import VitalsGeneratorV2

def generate_patient_specific_stable_sequence(profile_factors, duration=60) -> np.ndarray:
    """Generate normal vitals using VitalsGeneratorV2."""
    age = profile_factors.get('age', 30)
    gender = profile_factors.get('gender', 'male')
    generator = VitalsGeneratorV2(age=age, gender=gender, initial_state=0, duration=duration)
    sequence, _ = generator.generate_episode()
    return sequence[:duration, :]

def generate_patient_specific_emergency_sequence(profile_factors, emergency_type, duration=60) -> np.ndarray:
    """Generate emergency sequence using VitalsGeneratorV2."""
    age = profile_factors.get('age', 30)
    gender = profile_factors.get('gender', 'male')
    # initial_state can map to emergency types
    generator = VitalsGeneratorV2(age=age, gender=gender, initial_state=emergency_type, duration=duration)
    sequence, _ = generator.generate_episode()
    return sequence[:duration, :]

if __name__ == '__main__':
    print("--- Testing sim_gen.py functions ---")
    test_profile = {'age': 65, 'gender': 'male'}
    stable_seq = generate_patient_specific_stable_sequence(test_profile)
    print(f"Generated stable sequence shape: {stable_seq.shape}")
    emergency_seq = generate_patient_specific_emergency_sequence(test_profile, 1)
    print(f"Generated emergency sequence shape: {emergency_seq.shape}")
    print("--- sim_gen.py tests completed successfully ---")

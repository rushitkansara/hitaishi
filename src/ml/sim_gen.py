import numpy as np
import random
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import necessary components from datagen.py
from data_gen.datagen import (
    features, classes, PROFILE_FACTORS, noise_levels, valid_ranges,
    get_baselines, generate_monitor_sequence,
    generate_emergency_sequence, add_realistic_noise, clip_to_valid_ranges,
    apply_augmentation, emergency_scenarios
)

def generate_stable_sequence(baselines, duration=60):
    """Generate normal vitals using a bounded random walk for natural fluctuation."""
    sequence = np.zeros((duration, len(features)))

    # Configuration for the random walk of each vital
    walk_params = {
        'heart_rate':       {'bound': 4, 'max_step': 0.5},
        'systolic_bp':      {'bound': 5, 'max_step': 2},
        'diastolic_bp':     {'bound': 4, 'max_step': 2},
        'spo2':             {'bound': 1, 'max_step': 0.2},
        'temperature':      {'bound': 0.1, 'max_step': 0.1},
        'respiratory_rate': {'bound': 2, 'max_step': 1},
        'blood_glucose':    {'bound': 5, 'max_step': 3},
    }

    for i, feature_name in features.items():
        baseline_val = baselines[feature_name]
        params = walk_params[feature_name]
        min_bound = baseline_val - params['bound']
        max_bound = baseline_val + params['bound']
        max_step = params['max_step']

        # Start the sequence at the baseline
        sequence[0, i] = baseline_val

        for t in range(1, duration):
            prev_val = sequence[t-1, i]
            
            # Generate a random step
            step = random.uniform(-max_step, max_step)
            new_val = prev_val + step

            # If the new value exceeds the bounds, push it back
            if new_val > max_bound:
                new_val = max_bound - abs(random.uniform(0, max_step))
            elif new_val < min_bound:
                new_val = min_bound + abs(random.uniform(0, max_step))
            
            # Ensure value stays within globally valid physiological ranges as a final check
            valid_min, valid_max = valid_ranges[feature_name]
            sequence[t, i] = np.clip(new_val, valid_min, valid_max)
            
    return sequence

def generate_patient_specific_stable_sequence(profile_factors, duration=60):
    """Generates a 60-second stable vital sign sequence for a specific patient profile."""
    baselines = get_baselines(profile_factors)
    sequence = generate_stable_sequence(baselines, duration)
    sequence = clip_to_valid_ranges(sequence, valid_ranges)
    return sequence

def generate_patient_specific_emergency_sequence(profile_factors, emergency_type, duration=60):
    """Generates a 60-second emergency vital sign sequence for a specific patient profile and emergency type."""
    baselines = get_baselines(profile_factors)
    
    # Ensure emergency_type is a valid class label
    if emergency_type not in classes:
        raise ValueError(f"Invalid emergency_type: {emergency_type}. Must be one of {list(classes.keys())}")

    # Stable and Monitor cases are handled by generate_patient_specific_stable_sequence or generate_monitor_sequence
    if emergency_type == 0: # Stable
        sequence = generate_stable_sequence(baselines, duration)
    elif emergency_type == 1: # Monitor
        sequence = generate_monitor_sequence(baselines, duration)
    else: # Emergency cases
        scenario_variations = emergency_scenarios[emergency_type]
        sequence = generate_emergency_sequence(baselines, scenario_variations, duration)

    sequence = add_realistic_noise(sequence, noise_levels)
    
    # Apply augmentation to emergency sequences to add more variability
    method = random.choice(['time_warping', 'magnitude_scaling', 'gaussian_jitter', 'baseline_shift'])
    sequence = apply_augmentation(sequence, method)

    sequence = clip_to_valid_ranges(sequence, valid_ranges)
    return sequence

if __name__ == '__main__':
    print("--- Testing sim_gen.py functions ---")
    # Example profile factors
    test_profile = {
        'age': 65,
        'gender': 'male',
        'activity_level': 'sedentary',
        'primary_condition': 'Hypertension'
    }

    # Test stable sequence generation
    stable_seq = generate_patient_specific_stable_sequence(test_profile)
    print(f"Generated stable sequence shape: {stable_seq.shape}")
    assert stable_seq.shape == (60, len(features))
    for i, feature_name in features.items():
        min_val, max_val = valid_ranges[feature_name.lower()]
        assert stable_seq[:, i].min() >= min_val and stable_seq[:, i].max() <= max_val
    print("Stable sequence validation: PASSED")

    # Test emergency sequence generation (e.g., Heart Attack)
    emergency_seq = generate_patient_specific_emergency_sequence(test_profile, 2)
    print(f"Generated emergency sequence shape: {emergency_seq.shape}")
    assert emergency_seq.shape == (60, len(features))
    for i, feature_name in features.items():
        min_val, max_val = valid_ranges[feature_name.lower()]
        assert emergency_seq[:, i].min() >= min_val and emergency_seq[:, i].max() <= max_val
    print("Emergency sequence validation: PASSED")

    print("--- sim_gen.py tests completed successfully ---")

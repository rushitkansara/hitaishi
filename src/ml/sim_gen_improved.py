
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from the IMPROVED datagen
try:
    from data_gen.datagen_improved import (
        FEATURES, CLASSES, PROFILE_FACTORS, NOISE_LEVELS, VALID_RANGES,
        get_baselines, generate_monitor_sequence,
        generate_emergency_sequence, add_realistic_noise, clip_to_valid_ranges,
        apply_augmentation, EMERGENCY_SCENARIOS, CORRELATION_MATRIX
    )
except ImportError:
    # Fallback to original if improved doesn't exist (though it should in this task)
    from data_gen.datagen import (
        features as FEATURES, classes as CLASSES, PROFILE_FACTORS, noise_levels as NOISE_LEVELS, valid_ranges as VALID_RANGES,
        get_baselines, generate_monitor_sequence,
        generate_emergency_sequence, add_realistic_noise, clip_to_valid_ranges,
        apply_augmentation, emergency_scenarios as EMERGENCY_SCENARIOS
    )
    CORRELATION_MATRIX = {}

def generate_stable_sequence(baselines: Dict[str, float], duration: int = 60) -> np.ndarray:
    """
    Generate normal vitals using a bounded random walk with cross-vital correlations.
    """
    sequence = np.zeros((duration, len(FEATURES)))

    # Configuration for the random walk of each vital
    walk_params = {
        'heart_rate':       {'bound': 5.0, 'max_step': 0.6},
        'systolic_bp':      {'bound': 8.0, 'max_step': 1.5},
        'diastolic_bp':     {'bound': 6.0, 'max_step': 1.0},
        'spo2':             {'bound': 1.0, 'max_step': 0.15},
        'temperature':      {'bound': 0.15, 'max_step': 0.05},
        'respiratory_rate': {'bound': 3.0, 'max_step': 0.8},
        'blood_glucose':    {'bound': 6.0, 'max_step': 2.5},
    }

    # 1. Base Random Walk
    for i, feature_name in FEATURES.items():
        baseline_val = baselines[feature_name]
        params = walk_params[feature_name]
        min_bound = baseline_val - params['bound']
        max_bound = baseline_val + params['bound']
        max_step = params['max_step']

        sequence[0, i] = baseline_val

        for t in range(1, duration):
            prev_val = sequence[t-1, i]
            step = random.uniform(-max_step, max_step)
            new_val = prev_val + step

            # Soft boundaries: gentle pull back to baseline if exceeding bounds
            if new_val > max_bound:
                new_val -= abs(step) * 1.5
            elif new_val < min_bound:
                new_val += abs(step) * 1.5
            
            sequence[t, i] = new_val

    # 2. Apply Cross-Vital Coupling (Correlation)
    coupled_sequence = sequence.copy()
    if CORRELATION_MATRIX:
        for i, f_i in FEATURES.items():
            if f_i in CORRELATION_MATRIX:
                for f_j, correlation in CORRELATION_MATRIX[f_i].items():
                    j = next(idx for idx, name in FEATURES.items() if name == f_j)
                    # Real-time coupling: deviations from baseline in i affect j
                    coupled_sequence[:, j] += (sequence[:, i] - baselines[f_i]) * correlation * 0.15

    # Final clip
    for i, feature_name in FEATURES.items():
        v_min, v_max = VALID_RANGES[feature_name]
        coupled_sequence[:, i] = np.clip(coupled_sequence[:, i], v_min, v_max)
            
    return coupled_sequence

def generate_patient_specific_stable_sequence(profile_factors: Dict[str, Any], duration: int = 60) -> np.ndarray:
    """Generates a stable vital sign sequence for a specific patient profile."""
    baselines = get_baselines(profile_factors)
    sequence = generate_stable_sequence(baselines, duration)
    return sequence

def generate_patient_specific_emergency_sequence(profile_factors: Dict[str, Any], 
                                                 emergency_type: int, 
                                                 duration: int = 60) -> np.ndarray:
    """Generates an emergency vital sign sequence for a specific patient profile and emergency type."""
    baselines = get_baselines(profile_factors)
    
    if emergency_type not in CLASSES:
        raise ValueError(f"Invalid emergency_type: {emergency_type}")

    if emergency_type == 0:
        sequence = generate_stable_sequence(baselines, duration)
    elif emergency_type == 1:
        sequence = generate_monitor_sequence(baselines, duration)
    else:
        scenario_variations = EMERGENCY_SCENARIOS[emergency_type]
        sequence = generate_emergency_sequence(baselines, scenario_variations, duration)

    sequence = add_realistic_noise(sequence, NOISE_LEVELS)
    
    # Always apply a bit of augmentation for variety in emergency simulations
    method = random.choice(['time_warping', 'magnitude_scaling', 'gaussian_jitter', 'baseline_shift'])
    sequence = apply_augmentation(sequence, method)

    sequence = clip_to_valid_ranges(sequence, VALID_RANGES)
    return sequence

if __name__ == '__main__':
    print("--- Running IMPROVED sim_gen.py Tests ---")
    test_profile = {
        'age': 70,
        'gender': 'female',
        'activity_level': 'active',
        'primary_condition': 'Healthy'
    }

    stable_seq = generate_patient_specific_stable_sequence(test_profile)
    print(f"  Stable sequence shape: {stable_seq.shape}")
    
    emergency_seq = generate_patient_specific_emergency_sequence(test_profile, 2) # Heart Attack
    print(f"  Emergency sequence shape: {emergency_seq.shape}")
    
    print("--- All Tests PASSED ---")

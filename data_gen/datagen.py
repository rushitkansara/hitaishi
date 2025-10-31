import numpy as np
import pandas as pd
import json
import pickle
import random
from datetime import datetime
import argparse
from scipy.interpolate import interp1d
import textwrap
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config # Import config for data paths

# 1. DATA STRUCTURE
features = {
    0: 'heart_rate',
    1: 'systolic_bp',
    2: 'diastolic_bp',
    3: 'spo2',
    4: 'temperature',
    5: 'respiratory_rate',
    6: 'blood_glucose'
}

classes = {
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

# 2. PROCEDURAL PROFILE GENERATION FRAMEWORK

# These factors are the building blocks for creating diverse patient profiles.
PROFILE_FACTORS = {
    'genders': ['male', 'female'],
    'age_bins': [(18, 30), (31, 45), (46, 60), (61, 75), (76, 90)],
    'activity_levels': ['sedentary', 'moderate', 'active'],
    'primary_conditions': [
        'Healthy',
        'Hypertension',
        'Type 2 Diabetes',
        'Heart Failure',
        'COPD',
        'Obesity',
        'Chronic Kidney Disease',
        'Atrial Fibrillation'
    ]
}


# 3. TEMPORAL PROGRESSION LOGIC
emergency_scenarios = {
    2: { # Heart Attack
        'classic_stemi': {
            'phase_1': {'duration': 15, 'params': {
                'heart_rate': {'end': 25, 'pattern': 'linear'}, 'systolic_bp': {'end': 20, 'pattern': 'linear'},
                'respiratory_rate': {'end': 5, 'pattern': 'linear'}
            }},
            'phase_2': {'duration': 25, 'params': {
                'heart_rate': {'end': 45, 'pattern': 'spike', 'irregularity': 0.4}, 'systolic_bp': {'end': -30, 'pattern': 'plateau_drop'},
                'diastolic_bp': {'end': -15, 'pattern': 'linear'}, 'spo2': {'end': -5, 'pattern': 'linear'},
                'temperature': {'end': -0.5, 'pattern': 'linear'}
            }},
            'phase_3': {'duration': 20, 'params': {
                'heart_rate': {'end': 15, 'pattern': 'spike', 'irregularity': 0.6}, 'systolic_bp': {'end': -60, 'pattern': 'collapse'},
                'diastolic_bp': {'end': -30, 'pattern': 'collapse'}, 'spo2': {'end': -8, 'pattern': 'collapse'}
            }}
        },
        'slow_onset_nstemi': {
            'phase_1': {'duration': 30, 'params': {
                'heart_rate': {'end': 15, 'pattern': 'linear'}, 'systolic_bp': {'end': 10, 'pattern': 'linear'},
                'spo2': {'end': -2, 'pattern': 'linear'}
            }},
            'phase_2': {'duration': 30, 'params': {
                'heart_rate': {'end': 20, 'pattern': 'linear', 'irregularity': 0.2}, 'systolic_bp': {'end': -15, 'pattern': 'linear'},
                'diastolic_bp': {'end': -10, 'pattern': 'linear'}, 'spo2': {'end': -3, 'pattern': 'linear'}
            }}
        },
        'silent_mi_diabetic': {
            'phase_1': {'duration': 60, 'params': {
                'blood_glucose': {'end': 50, 'pattern': 'linear'}, 'spo2': {'end': -4, 'pattern': 'linear'},
                'respiratory_rate': {'end': 6, 'pattern': 'linear'}
            }}
        }
    },
    3: { # Arrhythmia
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'heart_rate': {'pattern': 'very_irregular', 'irregularity': 0.6, 'noise_std': 20}
            }}
        }
    },
    4: { # Heart Failure
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'heart_rate': {'end': 15, 'pattern': 'linear'}, 'systolic_bp': {'end': -20, 'pattern': 'linear'},
                'diastolic_bp': {'end': -10, 'pattern': 'linear'}, 'spo2': {'end': -5, 'pattern': 'linear'},
                'respiratory_rate': {'end': 8, 'pattern': 'linear'}
            }}
        }
    },
    5: { # Hypoglycemia
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'blood_glucose': {'end': -60, 'pattern': 'collapse'}, 'heart_rate': {'end': 30, 'pattern': 'linear'},
                'temperature': {'end': -0.8, 'pattern': 'linear'}, 'respiratory_rate': {'end': 5, 'pattern': 'linear'}
            }}
        }
    },
    6: { # Hyperglycemia_DKA
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'blood_glucose': {'end': 300, 'pattern': 'linear'}, 'heart_rate': {'end': 20, 'pattern': 'linear'},
                'respiratory_rate': {'end': 10, 'pattern': 'linear'}, 'systolic_bp': {'end': -10, 'pattern': 'linear'},
                'diastolic_bp': {'end': -5, 'pattern': 'linear'}
            }}
        }
    },
    7: { # Respiratory Distress
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'spo2': {'end': -15, 'pattern': 'collapse'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
                'heart_rate': {'end': 30, 'pattern': 'linear'}
            }}
        }
    },
    8: { # Sepsis
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'heart_rate': {'end': 40, 'pattern': 'linear'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
                'temperature': {'end': 2, 'pattern': 'linear'}, 'systolic_bp': {'end': -30, 'pattern': 'collapse'},
                'diastolic_bp': {'end': -15, 'pattern': 'collapse'}, 'spo2': {'end': -8, 'pattern': 'linear'}
            }}
        }
    },
    9: { # Stroke
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'systolic_bp': {'end': 80, 'pattern': 'spike'}, 'diastolic_bp': {'end': 40, 'pattern': 'spike'}
            }}
        }
    },
    10: { # Shock
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'heart_rate': {'end': 50, 'pattern': 'linear'}, 'systolic_bp': {'end': -50, 'pattern': 'collapse'},
                'diastolic_bp': {'end': -30, 'pattern': 'collapse'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
                'spo2': {'end': -10, 'pattern': 'linear'}, 'temperature': {'end': -1.5, 'pattern': 'linear'}
            }}
        }
    },
    11: { # Hypertensive Crisis
        'default': {
            'phase_1': {'duration': 60, 'params': {
                'systolic_bp': {'end': 100, 'pattern': 'spike'}, 'diastolic_bp': {'end': 50, 'pattern': 'spike'},
                'heart_rate': {'end': 10, 'pattern': 'linear'}, 'respiratory_rate': {'end': 5, 'pattern': 'linear'}
            }}
        }
    },
    12: { # Fall_Unconscious
        'default': {
            'phase_1': {'duration': 10, 'params': {}},
            'phase_2': {'duration': 50, 'params': {
                'heart_rate': {'end': -10, 'pattern': 'linear'},
                'systolic_bp': {'end': -10, 'pattern': 'linear'}, 'diastolic_bp': {'end': -5, 'pattern': 'linear'}
            }}
        }
    }
}


# 4. NOISE AND VARIABILITY REQUIREMENTS
noise_levels = {
    'heart_rate': 2.0,
    'systolic_bp': 5.0,
    'diastolic_bp': 3.0,
    'spo2': 0.5,
    'temperature': 0.1,
    'respiratory_rate': 1.0,
    'blood_glucose': 5.0
}

valid_ranges = {
    'heart_rate': (30, 220),
    'systolic_bp': (60, 250),
    'diastolic_bp': (40, 180),
    'spo2': (70, 100),
    'temperature': (34.0, 42.0), # In Celsius
    'respiratory_rate': (5, 60),
    'blood_glucose': (30, 600)
}

def get_baselines(profile_factors):
    """Procedurally calculates baseline vitals based on a profile's factors."""
    # Start with a baseline for a healthy, 30-year-old, moderately active male
    baselines = {
        'heart_rate': 70,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'spo2': 98,
        'temperature': 37.0, # Celsius
        'respiratory_rate': 16,
        'blood_glucose': 90
    }

    # Adjust for age
    age = profile_factors['age']
    if age > 80:
        baselines['heart_rate'] -= 5
        baselines['systolic_bp'] += 15
        baselines['diastolic_bp'] += 5
        baselines['spo2'] -= 1
    elif age > 60:
        baselines['heart_rate'] -= 2
        baselines['systolic_bp'] += 10
        baselines['diastolic_bp'] += 5
    elif age < 30:
        baselines['heart_rate'] += 2

    # Adjust for gender
    if profile_factors['gender'] == 'female':
        baselines['heart_rate'] += 5  # Women tend to have slightly higher resting heart rates

    # Adjust for activity level's influence on other vitals
    if profile_factors['activity_level'] == 'sedentary':
        baselines['heart_rate'] += 5
    elif profile_factors['activity_level'] == 'active':
        baselines['heart_rate'] -= 10

    # Adjust for primary condition
    condition = profile_factors['primary_condition']
    if condition == 'Hypertension':
        baselines['systolic_bp'] += 20
        baselines['diastolic_bp'] += 10
    elif condition == 'Type 2 Diabetes':
        baselines['blood_glucose'] = 150
        baselines['systolic_bp'] += 5
    elif condition == 'Heart Failure':
        baselines['heart_rate'] += 10
        baselines['systolic_bp'] -= 10
        baselines['spo2'] -= 2
    elif condition == 'COPD':
        baselines['spo2'] -= 4
        baselines['respiratory_rate'] += 4
    elif condition == 'Obesity':
        baselines['systolic_bp'] += 10
        baselines['diastolic_bp'] += 5
        baselines['heart_rate'] += 5
    elif condition == 'Chronic Kidney Disease':
        baselines['systolic_bp'] += 15
        baselines['diastolic_bp'] += 5
        baselines['spo2'] -= 1
    elif condition == 'Atrial Fibrillation':
        baselines['heart_rate'] += 15

    return baselines


def generate_stable_sequence(baselines, duration=60):
    """Generate normal vitals with natural, smoothly varying variations."""
    sequence = np.zeros((duration, len(features)))
    for i, feature_name in features.items():
        baseline_val = baselines[feature_name]
        # Create a few random anchor points for a smooth wander
        num_anchors = random.randint(4, 8)
        anchor_points = np.linspace(0, duration - 1, num_anchors, dtype=int)
        # Anchor values are small deviations from the baseline
        anchor_values = np.random.normal(baseline_val, noise_levels[feature_name.lower()], num_anchors)
        # Ensure the start and end are close to the baseline
        anchor_values[0] = baseline_val
        anchor_values[-1] = baseline_val

        # Interpolate between the anchor points to create a smooth signal
        f = interp1d(anchor_points, anchor_values, kind='cubic')
        sequence[:, i] = f(np.arange(duration))
    return sequence





def generate_monitor_sequence(baselines, duration=60):


    """Generate mildly abnormal vitals"""


    sequence = generate_stable_sequence(baselines, duration)


    # Introduce a mild, persistent abnormality


    param_to_alter = random.choice(['hr', 'systolic', 'resp', 'spo2'])


    if param_to_alter == 'hr':


        sequence[:, 0] += random.uniform(10, 20) # Slightly tachycardic


    elif param_to_alter == 'systolic':


        sequence[:, 1] += random.uniform(10, 20) # Slightly hypertensive


    elif param_to_alter == 'resp':


        sequence[:, 5] += random.uniform(4, 8) # Slightly tachypneic


    elif param_to_alter == 'spo2':


        sequence[:, 3] -= random.uniform(2, 4) # Mild desaturation


    return sequence





def generate_emergency_sequence(baselines, scenario_variations, duration=60):


    """Generate a sequence based on a randomly chosen variation of an emergency scenario."""


    # Randomly select one of the scenario variations


    variation_name = random.choice(list(scenario_variations.keys()))


    scenario_def = scenario_variations[variation_name]





    sequence = generate_stable_sequence(baselines, duration)


    current_time = 0

    for phase, phase_def in scenario_def.items():
        phase_duration = phase_def['duration']
        start_time = current_time
        end_time = start_time + phase_duration

        for i, param_name in features.items():
            if param_name in phase_def['params']:
                param_rules = phase_def['params'][param_name]
                start_val = sequence[start_time, i]
                end_val = start_val + param_rules.get('end', 0)
                pattern = param_rules.get('pattern', 'linear')
                
                x = np.linspace(0, 1, phase_duration)
                if pattern == 'linear':
                    values = start_val + x * (end_val - start_val)
                elif pattern == 'collapse':
                    values = start_val + (x**3) * (end_val - start_val)
                elif pattern == 'spike':
                    values = start_val + (1 - np.cos(x * np.pi)) / 2 * (end_val - start_val)
                elif pattern == 'plateau_drop':
                    values = np.full(phase_duration, start_val + (end_val - start_val) * 0.5)
                    if phase_duration > 10:
                        values[-10:] = np.linspace(values[-11], end_val, 10)
                elif pattern == 'immobile':
                    values = np.zeros(phase_duration)
                elif 'irregular' in pattern:
                    irregularity = param_rules.get('irregularity', 0.2)
                    noise_std = param_rules.get('noise_std', 5)
                    base_values = np.linspace(start_val, end_val, phase_duration)
                    jumps = np.random.randn(phase_duration) * noise_std * irregularity
                    values = base_values + np.cumsum(jumps)
                else:
                    values = np.linspace(start_val, end_val, phase_duration)

                sequence[start_time:end_time, i] = values

        current_time = end_time

    return sequence



def add_realistic_noise(sequence, noise_levels):
    """Add Gaussian noise to simulate sensor variability"""
    noisy_sequence = sequence.copy()
    for i, feature_name in features.items():
        noise = np.random.normal(0, noise_levels[feature_name.lower()], noisy_sequence.shape[0])
        noisy_sequence[:, i] += noise
    return noisy_sequence

def clip_to_valid_ranges(sequence, valid_ranges):
    """Ensure all values are physiologically possible"""
    clipped_sequence = sequence.copy()
    for i, feature_name in features.items():
        min_val, max_val = valid_ranges[feature_name.lower()]
        clipped_sequence[:, i] = np.clip(clipped_sequence[:, i], min_val, max_val)
    return clipped_sequence

def apply_augmentation(sequence, method):
    """Apply data augmentation technique"""
    if method == 'time_warping':
        scale = random.uniform(0.9, 1.1)
        augmented_sequence = np.zeros_like(sequence)
        original_indices = np.arange(sequence.shape[0])
        sampling_indices = np.linspace(0, sequence.shape[0] - 1, sequence.shape[0]) * scale
        sampling_indices = np.clip(sampling_indices, 0, sequence.shape[0] - 1)
        for i in range(sequence.shape[1]):
            original_signal = sequence[:, i]
            augmented_sequence[:, i] = np.interp(sampling_indices, original_indices, original_signal)
        return augmented_sequence
    elif method == 'magnitude_scaling':
        scale = random.uniform(0.95, 1.05)
        return sequence * scale
    elif method == 'gaussian_jitter':
        noise = np.random.normal(0, np.std(sequence) * 0.1, sequence.shape)
        return sequence + noise
    elif method == 'baseline_shift':
        shift = np.random.uniform(-0.05, 0.05, sequence.shape[1])
        return sequence + shift
    return sequence

def generate_procedural_dataset(samples_per_cohort_class, test_split_ratio=0.3):
    """Generates a balanced, procedural dataset with a clean train/test split."""
    X_train, y_train, X_test, y_test = [], [], [], []

    # Create all possible cohort combinations
    cohort_factors = []
    for gender in PROFILE_FACTORS['genders']:
        for age_bin in PROFILE_FACTORS['age_bins']:
            for activity in PROFILE_FACTORS['activity_levels']:
                for condition in PROFILE_FACTORS['primary_conditions']:
                    cohort_factors.append({
                        'gender': gender,
                        'age_bin': age_bin,
                        'activity_level': activity,
                        'primary_condition': condition
                    })

    # Split cohorts into training and testing groups
    random.shuffle(cohort_factors)
    split_index = int(len(cohort_factors) * (1 - test_split_ratio))
    train_cohorts = cohort_factors[:split_index]
    test_cohorts = cohort_factors[split_index:]

    print(f"Total cohorts: {len(cohort_factors)}")
    print(f"Training cohorts: {len(train_cohorts)}")
    print(f"Testing cohorts: {len(test_cohorts)}")

    # Generate data for training and testing sets
    for cohort_group, X_set, y_set, set_name in [(train_cohorts, X_train, y_train, 'Train'), (test_cohorts, X_test, y_test, 'Test')]:
        print(f"\n--- Generating {set_name} Set ---")
        for i, cohort in enumerate(cohort_group):
            print(f"  Processing {set_name} Cohort {i+1}/{len(cohort_group)}...")
            # For each cohort, generate data for all 13 classes
            for class_label in classes.keys():
                for _ in range(samples_per_cohort_class):
                    # Create a specific patient profile from the cohort
                    profile = {
                        'age': random.uniform(cohort['age_bin'][0], cohort['age_bin'][1]),
                        'gender': cohort['gender'],
                        'activity_level': cohort['activity_level'],
                        'primary_condition': cohort['primary_condition']
                    }
                    baselines = get_baselines(profile)
                    
                    # Generate sequence based on class
                    if class_label == 0:
                        sequence = generate_stable_sequence(baselines)
                    elif class_label == 1:
                        sequence = generate_monitor_sequence(baselines)
                    else:
                        scenario_variations = emergency_scenarios[class_label]
                        sequence = generate_emergency_sequence(baselines, scenario_variations)

                    sequence = add_realistic_noise(sequence, noise_levels)

                    # Apply augmentation to non-stable cases
                    if class_label != 0:
                        method = random.choice(['time_warping', 'magnitude_scaling', 'gaussian_jitter', 'baseline_shift'])
                        sequence = apply_augmentation(sequence, method)

                    sequence = clip_to_valid_ranges(sequence, valid_ranges)

                    X_set.append(sequence)
                    y_set.append(class_label)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

import textwrap

def save_outputs(X_train, y_train, X_test, y_test, output_dir='data/', seed=None):
    """Save all required output files for the new procedural dataset."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save datasets
    np.savez_compressed(os.path.join(output_dir, 'procedural_X_train.npz'), X_train=X_train)
    np.save(os.path.join(output_dir, 'procedural_y_train.npy'), y_train)
    np.savez_compressed(os.path.join(output_dir, 'procedural_X_test.npz'), X_test=X_test)
    np.save(os.path.join(output_dir, 'procedural_y_test.npy'), y_test)

    # Save scaler fit on training data only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if len(X_train) > 0:
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
    with open(os.path.join(output_dir, 'procedural_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Create the generation report string
    report_header = f"""
    Procedural Data Generation Report
    =================================
    Generation Timestamp: {datetime.now()}
    Random Seed Used: {seed}

    Total Training Samples: {len(y_train)}
    Total Testing Samples: {len(y_test)}
    """

    train_counts_header = "\nSample Counts per Class (Training):\n"
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    train_counts_lines = [f"  - Class {label} ({classes[label]}): {count}" for label, count in zip(unique_train, counts_train)]
    train_counts = train_counts_header + "\n".join(train_counts_lines)

    test_counts_header = "\n\nSample Counts per Class (Testing):\n"
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_counts_lines = [f"  - Class {label} ({classes[label]}): {count}" for label, count in zip(unique_test, counts_test)]
    test_counts = test_counts_header + "\n".join(test_counts_lines)

    full_report = textwrap.dedent(report_header) + textwrap.dedent(train_counts) + textwrap.dedent(test_counts)

    # Save generation report
    with open(os.path.join(output_dir, 'procedural_generation_report.txt'), 'w') as f:
        f.write(full_report)

    print(f"All procedural output files saved to '{output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate procedural synthetic medical time-series data.')
    parser.add_argument('--samples_per_cohort', type=int, default=20, help='Number of samples to generate per cohort and class combination.')
    parser.add_argument('--output', type=str, default=config.DATA_DIR, help='Output directory for generated files.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train, X_test, y_test = generate_procedural_dataset(args.samples_per_cohort)
    save_outputs(X_train, y_train, X_test, y_test, args.output, seed=args.seed)

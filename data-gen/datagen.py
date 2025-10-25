import numpy as np
import pandas as pd
import json
import pickle
import random
from datetime import datetime
import argparse
from scipy.interpolate import interp1d
import textwrap

# 1. DATA STRUCTURE
features = {
    0: 'heart_rate',
    1: 'systolic_bp',
    2: 'diastolic_bp',
    3: 'spo2',
    4: 'temperature',
    5: 'respiratory_rate',
    6: 'blood_glucose',
    7: 'activity_level'
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

# 2. PATIENT PROFILE SYSTEM
profiles = {
    1: {
        'age': 65, 'gender': 'male',
        'conditions': ['diabetes_type2', 'hypertension', 'previous_mi'],
        'baselines': {
            'heart_rate': 58, 'systolic_bp': 140, 'diastolic_bp': 85,
            'spo2': 97, 'temperature': 36.8, 'respiratory_rate': 16,
            'blood_glucose': 180, 'activity_level': 30
        }
    },
    2: {
        'age': 34, 'gender': 'female',
        'conditions': ['gestational_diabetes', 'pregnancy_week_28'],
        'baselines': {
            'heart_rate': 88, 'systolic_bp': 125, 'diastolic_bp': 80,
            'spo2': 98, 'temperature': 36.9, 'respiratory_rate': 18,
            'blood_glucose': 145, 'activity_level': 40
        }
    },
    3: {
        'age': 45, 'gender': 'male',
        'conditions': ['obesity_bmi_35', 'sleep_apnea', 'copd'],
        'baselines': {
            'heart_rate': 72, 'systolic_bp': 135, 'diastolic_bp': 88,
            'spo2': 94, 'temperature': 36.7, 'respiratory_rate': 18,
            'blood_glucose': 110, 'activity_level': 20
        }
    },
    4: {
        'age': 78, 'gender': 'female',
        'conditions': ['atrial_fibrillation', 'heart_failure', 'stroke_history'],
        'baselines': {
            'heart_rate': 65, 'systolic_bp': 110, 'diastolic_bp': 70,
            'spo2': 96, 'temperature': 36.5, 'respiratory_rate': 17,
            'blood_glucose': 95, 'activity_level': 15
        }
    },
    5: {
        'age': 28, 'gender': 'male',
        'conditions': ['athletic_heart', 'baseline_bradycardia'],
        'baselines': {
            'heart_rate': 48, 'systolic_bp': 105, 'diastolic_bp': 65,
            'spo2': 99, 'temperature': 36.6, 'respiratory_rate': 14,
            'blood_glucose': 85, 'activity_level': 60
        }
    }
}


# 3. TEMPORAL PROGRESSION LOGIC
emergency_scenarios = {
    2: { # Heart Attack
        'phase_1': {'duration': 20, 'params': {
            'heart_rate': {'end': 20, 'pattern': 'linear'}, 'systolic_bp': {'end': 20, 'pattern': 'linear'},
            'diastolic_bp': {'end': 10, 'pattern': 'linear'}, 'spo2': {'end': -2, 'pattern': 'linear'},
            'respiratory_rate': {'end': 4, 'pattern': 'linear'}, 'activity_level': {'end': -20, 'pattern': 'linear'}
        }},
        'phase_2': {'duration': 20, 'params': {
            'heart_rate': {'end': 40, 'pattern': 'spike', 'irregularity': 0.3}, 'systolic_bp': {'end': -10, 'pattern': 'plateau_drop'},
            'diastolic_bp': {'end': -5, 'pattern': 'linear'}, 'spo2': {'end': -4, 'pattern': 'linear'},
            'temperature': {'end': 0.4, 'pattern': 'linear'}, 'respiratory_rate': {'end': 4, 'pattern': 'linear'},
            'activity_level': {'end': -8, 'pattern': 'linear'}
        }},
        'phase_3': {'duration': 20, 'params': {
            'heart_rate': {'end': 10, 'pattern': 'spike', 'irregularity': 0.5}, 'systolic_bp': {'end': -50, 'pattern': 'collapse'},
            'diastolic_bp': {'end': -25, 'pattern': 'collapse'}, 'spo2': {'end': -4, 'pattern': 'collapse'},
            'temperature': {'end': 0.3, 'pattern': 'linear'}, 'respiratory_rate': {'end': 4, 'pattern': 'linear'},
            'activity_level': {'end': -2, 'pattern': 'immobile'}
        }}
    },
    3: { # Arrhythmia
        'phase_1': {'duration': 60, 'params': {
            'heart_rate': {'pattern': 'very_irregular', 'irregularity': 0.6, 'noise_std': 20}
        }}
    },
    4: { # Heart Failure
        'phase_1': {'duration': 60, 'params': {
            'heart_rate': {'end': 15, 'pattern': 'linear'}, 'systolic_bp': {'end': -20, 'pattern': 'linear'},
            'diastolic_bp': {'end': -10, 'pattern': 'linear'}, 'spo2': {'end': -5, 'pattern': 'linear'},
            'respiratory_rate': {'end': 8, 'pattern': 'linear'}, 'activity_level': {'end': -15, 'pattern': 'linear'}
        }}
    },
    5: { # Hypoglycemia
        'phase_1': {'duration': 60, 'params': {
            'blood_glucose': {'end': -60, 'pattern': 'collapse'}, 'heart_rate': {'end': 30, 'pattern': 'linear'},
            'temperature': {'end': -0.8, 'pattern': 'linear'}, 'respiratory_rate': {'end': 5, 'pattern': 'linear'},
            'activity_level': {'end': -20, 'pattern': 'linear'}
        }}
    },
    6: { # Hyperglycemia_DKA
        'phase_1': {'duration': 60, 'params': {
            'blood_glucose': {'end': 300, 'pattern': 'linear'}, 'heart_rate': {'end': 20, 'pattern': 'linear'},
            'respiratory_rate': {'end': 10, 'pattern': 'linear'}, 'systolic_bp': {'end': -10, 'pattern': 'linear'},
            'diastolic_bp': {'end': -5, 'pattern': 'linear'}, 'activity_level': {'end': -15, 'pattern': 'linear'}
        }}
    },
    7: { # Respiratory Distress
        'phase_1': {'duration': 60, 'params': {
            'spo2': {'end': -15, 'pattern': 'collapse'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
            'heart_rate': {'end': 30, 'pattern': 'linear'}, 'activity_level': {'end': -30, 'pattern': 'immobile'}
        }}
    },
    8: { # Sepsis
        'phase_1': {'duration': 60, 'params': {
            'heart_rate': {'end': 40, 'pattern': 'linear'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
            'temperature': {'end': 2, 'pattern': 'linear'}, 'systolic_bp': {'end': -30, 'pattern': 'collapse'},
            'diastolic_bp': {'end': -15, 'pattern': 'collapse'}, 'spo2': {'end': -8, 'pattern': 'linear'},
            'activity_level': {'end': -30, 'pattern': 'immobile'}
        }}
    },
    9: { # Stroke
        'phase_1': {'duration': 60, 'params': {
            'systolic_bp': {'end': 80, 'pattern': 'spike'}, 'diastolic_bp': {'end': 40, 'pattern': 'spike'},
            'activity_level': {'end': -30, 'pattern': 'immobile'}
        }}
    },
    10: { # Shock
        'phase_1': {'duration': 60, 'params': {
            'heart_rate': {'end': 50, 'pattern': 'linear'}, 'systolic_bp': {'end': -50, 'pattern': 'collapse'},
            'diastolic_bp': {'end': -30, 'pattern': 'collapse'}, 'respiratory_rate': {'end': 20, 'pattern': 'linear'},
            'spo2': {'end': -10, 'pattern': 'linear'}, 'temperature': {'end': -1.5, 'pattern': 'linear'},
            'activity_level': {'end': -30, 'pattern': 'immobile'}
        }}
    },
    11: { # Hypertensive Crisis
        'phase_1': {'duration': 60, 'params': {
            'systolic_bp': {'end': 100, 'pattern': 'spike'}, 'diastolic_bp': {'end': 50, 'pattern': 'spike'},
            'heart_rate': {'end': 10, 'pattern': 'linear'}, 'respiratory_rate': {'end': 5, 'pattern': 'linear'},
            'activity_level': {'end': -20, 'pattern': 'linear'}
        }}
    },
    12: { # Fall_Unconscious
        'phase_1': {'duration': 10, 'params': {'activity_level': {'end': -30, 'pattern': 'collapse'}}},
        'phase_2': {'duration': 50, 'params': {
            'activity_level': {'pattern': 'immobile'}, 'heart_rate': {'end': -10, 'pattern': 'linear'},
            'systolic_bp': {'end': -10, 'pattern': 'linear'}, 'diastolic_bp': {'end': -5, 'pattern': 'linear'}
        }}
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
    'blood_glucose': 5.0,
    'activity_level': 10.0
}

valid_ranges = {
    'heart_rate': (30, 220),
    'systolic_bp': (60, 250),
    'diastolic_bp': (40, 180),
    'spo2': (70, 100),
    'temperature': (34.0, 42.0),
    'respiratory_rate': (5, 60),
    'blood_glucose': (30, 600),
    'activity_level': (0, 200)
}

# 8. CODE STRUCTURE REQUIREMENTS
class PatientProfile:
    """Store baseline vitals and medical conditions"""
    def __init__(self, profile_id):
        profile_data = profiles[profile_id]
        self.age = profile_data['age']
        self.gender = profile_data['gender']
        self.conditions = profile_data['conditions']
        self.baselines = profile_data['baselines']

class EmergencyScenario:
    """Define temporal progression patterns for each medical case"""
    pass


def generate_stable_sequence(profile, duration=60):
    """Generate normal vitals with natural variations"""
    sequence = np.zeros((duration, len(features)))
    baselines = profile.baselines

    for i, feature_name in features.items():
        sequence[:, i] = baselines[feature_name]
    
    return sequence


def generate_monitor_sequence(profile, duration=60):
    """Generate mildly abnormal vitals"""
    sequence = generate_stable_sequence(profile, duration)
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

def generate_emergency_sequence(profile, scenario_def, duration=60):
    """Generate a sequence based on a multi-phase emergency scenario."""
    sequence = generate_stable_sequence(profile, duration)
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



class_distribution = {
    0: 1500, 1: 1000, 2: 400, 3: 300, 4: 250, 5: 400,
    6: 250, 7: 300, 8: 250, 9: 150, 10: 100, 11: 100, 12: 100
}

def add_realistic_noise(sequence, noise_levels):
    """Add Gaussian noise to simulate sensor variability"""
    noisy_sequence = sequence.copy()
    for i, feature_name in features.items():
        # Use feature_name.lower() to match keys in noise_levels
        noise = np.random.normal(0, noise_levels[feature_name.lower()], noisy_sequence.shape[0])
        noisy_sequence[:, i] += noise
    return noisy_sequence

def clip_to_valid_ranges(sequence, valid_ranges):
    """Ensure all values are physiologically possible"""
    clipped_sequence = sequence.copy()
    for i, feature_name in features.items():
        # Use feature_name.lower() to match keys in valid_ranges
        min_val, max_val = valid_ranges[feature_name.lower()]
        clipped_sequence[:, i] = np.clip(clipped_sequence[:, i], min_val, max_val)
    return clipped_sequence

def apply_augmentation(sequence, method):
    """Apply data augmentation technique"""
    if method == 'time_warping':
        scale = random.uniform(0.9, 1.1)
        augmented_sequence = np.zeros_like(sequence)
        
        original_indices = np.arange(sequence.shape[0])
        # Create a new set of indices to sample from the original signal.
        # Scaling the timeline achieves the time-warping effect.
        sampling_indices = np.linspace(0, sequence.shape[0] - 1, sequence.shape[0]) * scale
        # Clamp the indices to the valid range [0, 59] to prevent out-of-bounds issues, 
        # letting interp handle edges gracefully.
        sampling_indices = np.clip(sampling_indices, 0, sequence.shape[0] - 1)

        for i in range(sequence.shape[1]):
            original_signal = sequence[:, i]
            augmented_sequence[:, i] = np.interp(sampling_indices, original_indices, original_signal)
            
        return augmented_sequence
    elif method == 'magnitude_scaling':
        scale = random.uniform(0.95, 1.05)
        return sequence * scale
    elif method == 'gaussian_jitter':
        # Use a fraction of the standard deviation of the signal itself for more realistic noise
        noise = np.random.normal(0, np.std(sequence) * 0.1, sequence.shape)
        return sequence + noise
    elif method == 'baseline_shift':
        shift = np.random.uniform(-0.05, 0.05, sequence.shape[1])
        return sequence + shift
    return sequence

def generate_dataset(total_samples, noise_levels, valid_ranges):
    """Main generation loop"""
    X = []
    y = []

    for class_label, count in class_distribution.items():
        print(f"Generating {count} samples for class {class_label} ({classes[class_label]})")
        for _ in range(count):
            profile_id = random.choice(list(profiles.keys()))
            profile = PatientProfile(profile_id)
            
            if class_label == 0:
                sequence = generate_stable_sequence(profile)
            elif class_label == 1:
                sequence = generate_monitor_sequence(profile)
            else:
                scenario_def = emergency_scenarios[class_label]
                sequence = generate_emergency_sequence(profile, scenario_def)

            sequence = add_realistic_noise(sequence, noise_levels)
            sequence = clip_to_valid_ranges(sequence, valid_ranges)
            
            X.append(sequence)
            y.append(class_label)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def augment_dataset(X_train, y_train, n_augment):
    """Generate augmented samples from the training set."""
    X_aug = []
    y_aug = []
    augmentation_methods = ['time_warping', 'magnitude_scaling', 'gaussian_jitter', 'baseline_shift']
    
    if n_augment == 0:
        return np.array([]), np.array([])

    print(f"Generating {n_augment} augmented samples...")
    for i in range(n_augment):
        # Select a random sample to augment (excluding stable class)
        sample_idx = random.choice(np.where(y_train != 0)[0])
        sequence_to_augment = X_train[sample_idx]
        label = y_train[sample_idx]
        
        method = random.choice(augmentation_methods)
        augmented_sequence = apply_augmentation(sequence_to_augment, method)
        
        X_aug.append(augmented_sequence)
        y_aug.append(label)
        
    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)

import textwrap

def save_outputs(X_train, y_train, X_aug, y_aug, output_dir='data/', seed=None):
    """Save all required output files"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save datasets
    np.savez_compressed(os.path.join(output_dir, 'training_sequences.npz'), X_train=X_train)
    np.save(os.path.join(output_dir, 'training_labels.npy'), y_train)
    if X_aug.size > 0:
        np.savez_compressed(os.path.join(output_dir, 'augmented_sequences.npz'), X_aug=X_aug)
        np.save(os.path.join(output_dir, 'augmented_labels.npy'), y_aug)

    # Save patient profiles
    with open(os.path.join(output_dir, 'patient_profiles.json'), 'w') as f:
        json.dump(profiles, f, indent=4)

    # Save scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    with open(os.path.join(output_dir, 'scaler_params.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Create the generation report string
    report_header = f"""
    Data Generation Report
    ========================
    Generation Timestamp: {datetime.now()}
    Random Seed Used: {seed}

    Total Samples Generated: {len(y_train)}
    Total Samples Augmented: {len(y_aug)}
    """

    class_counts_header = "\nSample Counts per Class (Training):\n"
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts_lines = [f"  - Class {label} ({classes[label]}): {count}" for label, count in zip(unique, counts)]
    class_counts = class_counts_header + "\n".join(class_counts_lines)

    value_ranges_header = "\n\nValue Ranges per Parameter (Training Data):\n"
    value_ranges_lines = [f"  - {feature}: Min={X_train[:, :, i].min():.2f}, Max={X_train[:, :, i].max():.2f}, Mean={X_train[:, :, i].mean():.2f}" for i, feature in features.items()]
    value_ranges = value_ranges_header + "\n".join(value_ranges_lines)

    full_report = textwrap.dedent(report_header) + textwrap.dedent(class_counts) + textwrap.dedent(value_ranges)

    # Save generation report
    with open(os.path.join(output_dir, 'generation_report.txt'), 'w') as f:
        f.write(full_report)

    print(f"All output files saved to '{output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic medical time-series data.')
    parser.add_argument('--samples', type=int, default=5100, help='Number of primary samples to generate.')
    parser.add_argument('--augment', type=int, default=2000, help='Number of augmented samples to generate.')
    parser.add_argument('--output', type=str, default='data/', help='Output directory for generated files.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train = generate_dataset(args.samples, noise_levels, valid_ranges)
    X_aug, y_aug = augment_dataset(X_train, y_train, args.augment)
    save_outputs(X_train, y_train, X_aug, y_aug, args.output, seed=args.seed)
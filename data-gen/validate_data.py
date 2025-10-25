
import numpy as np
import argparse
import os
import json

# Import variables from the data generation script
from datagen import classes, features, valid_ranges, class_distribution

def validate_data(input_dir='data/'):
    """Validate the generated dataset based on the prompt's requirements."""
    print("--- Running Data Validation ---")
    
    # 1. File Presence Check
    required_files = [
        'training_sequences.npz', 'training_labels.npy',
        'augmented_sequences.npz', 'augmented_labels.npy',
        'patient_profiles.json', 'scaler_params.pkl', 'generation_report.txt'
    ]
    for f in required_files:
        assert os.path.exists(os.path.join(input_dir, f)), f"File not found: {f}"
    print("File presence check: PASSED")

    # 2. Load Data
    X_train = np.load(os.path.join(input_dir, 'training_sequences.npz'))['X_train']
    y_train = np.load(os.path.join(input_dir, 'training_labels.npy'))
    X_aug = np.load(os.path.join(input_dir, 'augmented_sequences.npz'))['X_aug']
    y_aug = np.load(os.path.join(input_dir, 'augmented_labels.npy'))

    # 3. Shape Validation
    expected_train_samples = sum(class_distribution.values())
    expected_aug_samples = 2000 # As per prompt
    assert X_train.shape == (expected_train_samples, 60, 8), f"X_train shape is {X_train.shape}, expected ({expected_train_samples}, 60, 8)"
    assert y_train.shape == (expected_train_samples,), f"y_train shape is {y_train.shape}, expected ({expected_train_samples},)"
    assert X_aug.shape == (expected_aug_samples, 60, 8), f"X_aug shape is {X_aug.shape}, expected ({expected_aug_samples}, 60, 8)"
    assert y_aug.shape == (expected_aug_samples,), f"y_aug shape is {y_aug.shape}, expected ({expected_aug_samples},)"
    print("Shape validation: PASSED")

    # 4. Label Validation
    assert np.all(np.isin(y_train, list(classes.keys()))), "y_train contains invalid labels"
    assert np.all(np.isin(y_aug, list(classes.keys()))), "y_aug contains invalid labels"
    print("Label validation: PASSED")

    # 5. Range Validation
    for i, feature_name in features.items():
        min_val, max_val = valid_ranges[feature_name.lower()]
        assert X_train[:, :, i].min() >= min_val, f"{feature_name} min value ({X_train[:, :, i].min()}) is out of range (< {min_val})"
        assert X_train[:, :, i].max() <= max_val, f"{feature_name} max value ({X_train[:, :, i].max()}) is out of range (> {max_val})"
    print("Range validation: PASSED")

    # 6. Class Balance Validation
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        expected_count = class_distribution[label]
        # Allow a small tolerance (e.g., 5%) in sample counts
        assert abs(count - expected_count) / expected_count <= 0.05, f"Class {label} has {count} samples, expected around {expected_count}"
    print("Class balance validation: PASSED")

    # 7. Temporal Coherence (No impossible jumps)
    diffs = np.diff(X_train, axis=1)
    # Check for jumps greater than 100, which would be highly unlikely in one second for most vitals
    assert not np.any(np.abs(diffs) > 100), "Temporal coherence check failed: Impossible jump detected"
    print("Temporal coherence validation: PASSED")

    # 8. Diversity Check (Std Dev)
    for i, feature_name in features.items():
        # Ensure there is some variation in the data, not just flat lines
        assert np.std(X_train[:, :, i]) > 0.1, f"Diversity check for {feature_name} failed (Std Dev is too low)"
    print("Diversity validation: PASSED")

    print("\n--- Validation Complete: All checks passed! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate synthetic medical time-series data.')
    parser.add_argument('--input', type=str, default='data/', help='Input directory containing the generated files.')
    args = parser.parse_args()

    validate_data(args.input)


import numpy as np
import argparse
import os
import json
import pandas as pd
from statsmodels.tsa.stattools import acf

# Import variables from the data generation script
from datagen import classes, features, valid_ranges

def validate_data(input_dir='data_gen/data/'):
    """Validate the new procedural dataset with statistical and clinical checks."""
    print("--- Running Procedural Data Validation ---")
    
    # 1. File Presence Check
    required_files = [
        'procedural_X_train.npz', 'procedural_y_train.npy',
        'procedural_X_test.npz', 'procedural_y_test.npy',
        'procedural_scaler.pkl', 'procedural_generation_report.txt'
    ]
    for f in required_files:
        assert os.path.exists(os.path.join(input_dir, f)), f"File not found: {f}"
    print("\n1. File presence check: PASSED")

    # 2. Load Data
    X_train = np.load(os.path.join(input_dir, 'procedural_X_train.npz'))['X_train']
    y_train = np.load(os.path.join(input_dir, 'procedural_y_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'procedural_X_test.npz'))['X_test']
    y_test = np.load(os.path.join(input_dir, 'procedural_y_test.npy'))
    print("\n2. Data loading: COMPLETED")

    # 3. Shape and Type Validation
    assert X_train.ndim == 3 and X_train.shape[2] == 7
    assert y_train.ndim == 1
    assert X_test.ndim == 3 and X_test.shape[2] == 7
    assert y_test.ndim == 1
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    print("\n3. Shape and type validation: PASSED")

    # 4. Range Validation
    for i, feature_name in features.items():
        min_val, max_val = valid_ranges[feature_name.lower()]
        assert X_train[:, :, i].min() >= min_val and X_train[:, :, i].max() <= max_val
        assert X_test[:, :, i].min() >= min_val and X_test[:, :, i].max() <= max_val
    print("\n4. Physiological range validation: PASSED")

    # 5. Statistical Analysis
    print("\n5. Statistical Analysis:")
    for class_label, class_name in classes.items():
        print(f"\n  --- Class: {class_name} ---")
        class_indices = np.where(y_train == class_label)[0]
        if len(class_indices) == 0:
            print("    No samples found in training set.")
            continue
        
        class_data = X_train[class_indices]
        
        # Distribution Summary
        print("    Distribution Summary (Mean, Std, Median):")
        for i, feature_name in features.items():
            feature_data = class_data[:, :, i].flatten()
            print(f"      - {feature_name:<15}: {np.mean(feature_data):>6.2f}, {np.std(feature_data):>6.2f}, {np.median(feature_data):>6.2f}")

        # Autocorrelation Check (for heart rate)
        hr_data = class_data[:, :, 0]
        autocorrelations = [acf(series, nlags=1, fft=False)[1] for series in hr_data]
        avg_acf = np.mean(autocorrelations)
        print(f"    Avg. Heart Rate Autocorrelation (Lag 1): {avg_acf:.2f}")
        assert avg_acf > 0.5, "Autocorrelation for HR is too low, signal may be too random."

    print("  Statistical analysis: COMPLETED")

    # 6. Clinical Guideline Checks
    print("\n6. Clinical Guideline Adherence Checks:")
    # Check Hypertensive Crisis
    ht_crisis_indices = np.where(y_train == 11)[0]
    if len(ht_crisis_indices) > 0:
        ht_crisis_data = X_train[ht_crisis_indices]
        max_systolic = np.max(ht_crisis_data[:, :, 1])
        print(f"  - Hypertensive Crisis: Max systolic BP reached: {max_systolic:.2f} (Threshold: >180)")
        assert max_systolic > 180, "Hypertensive Crisis data did not meet clinical threshold."

    # Check Hypoglycemia
    hypo_indices = np.where(y_train == 5)[0]
    if len(hypo_indices) > 0:
        hypo_data = X_train[hypo_indices]
        min_glucose = np.min(hypo_data[:, :, 6])
        print(f"  - Hypoglycemia: Min blood glucose reached: {min_glucose:.2f} (Threshold: <70)")
        assert min_glucose < 70, "Hypoglycemia data did not meet clinical threshold."
    print("  Clinical guideline checks: PASSED")

    print("\n--- Validation Complete: All checks passed! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate the procedural synthetic medical time-series data.')
    parser.add_argument('--input', type=str, default='data_gen/data/', help='Input directory for generated files.')
    args = parser.parse_args()

    validate_data(args.input)

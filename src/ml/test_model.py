
import numpy as np
import os
import pytest

# This assumes inference_engine.py is in the same directory
from inference_engine import HealthRiskPredictor

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Define paths at the top level
MODEL_PATH = config.HEALTH_LSTM_MODEL_PATH # Use the unquantized model
SCALER_PATH = config.SCALER_PARAMS_PATH
X_TEST_PATH = config.X_TEST_PATH
Y_TEST_PATH = config.Y_TEST_PATH

@pytest.fixture(scope="module")
def predictor():
    """Pytest fixture to initialize the HealthRiskPredictor once per module."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        pytest.fail("Model or scaler file not found. Run training first.")
    return HealthRiskPredictor(MODEL_PATH, SCALER_PATH)

@pytest.fixture(scope="module")
def test_data():
    """Pytest fixture to load test data once per module."""
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        pytest.fail("Test data not found. Run training script first.")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    return X_test, y_test

def test_predictor_initialization(predictor):
    """Tests if the predictor and its components are initialized correctly."""
    assert predictor.model is not None # Changed from interpreter to model
    assert predictor.scaler is not None
    assert len(predictor.class_names) == 13

def test_prediction_accuracy(predictor, test_data):
    """
    Tests the model's prediction accuracy on the test set.
    This is a sanity check and not a full evaluation.
    The target is a modest >80% accuracy on the quantized model.
    """
    X_test, y_test = test_data
    
    # Test on a subset to keep tests fast
    num_samples = min(100, len(X_test))
    X_sample = X_test[:num_samples]
    y_sample_indices = y_test[:num_samples]
    
    correct_predictions = 0
    for i in range(num_samples):
        risk_level, _, _, _ = predictor.predict(X_sample[i])
        if risk_level == y_sample_indices[i]:
            correct_predictions += 1
            
    accuracy = correct_predictions / num_samples
    print(f"Accuracy on {num_samples} test samples: {accuracy:.2%}")
    assert accuracy > 0.80 # Set a reasonable threshold for the test to pass

def test_edge_case_zero_input(predictor):
    """Tests the model's behavior with an input sequence of all zeros."""
    zero_seq = np.zeros((60, 8))
    try:
        risk_level, confidence, risk_name, _ = predictor.predict(zero_seq)
        # We expect a prediction, likely 'Stable'
        assert isinstance(risk_level, int)
        assert isinstance(confidence, float)
        assert isinstance(risk_name, str)
        print("✓ Handles zero input")
    except Exception as e:
        pytest.fail(f"Predictor failed on zero input: {e}")

def test_edge_case_extreme_values(predictor):
    """Tests the model's behavior with an input sequence of large values."""
    extreme_seq = np.full((60, 8), 1000.0)
    try:
        risk_level, confidence, risk_name, _ = predictor.predict(extreme_seq)
        assert isinstance(risk_level, int)
        assert isinstance(confidence, float)
        assert isinstance(risk_name, str)
        print("✓ Handles extreme values")
    except Exception as e:
        pytest.fail(f"Predictor failed on extreme values input: {e}")

def test_invalid_input_shape(predictor):
    """Tests that the predictor raises a ValueError for incorrect input shapes."""
    with pytest.raises(ValueError):
        invalid_seq = np.zeros((59, 8)) # Incorrect timesteps
        predictor.predict(invalid_seq)
        
    with pytest.raises(ValueError):
        invalid_seq = np.zeros((60, 7)) # Incorrect features
        predictor.predict(invalid_seq)

# This allows running the test file directly with `python src/ml/test_model.py`
# although `pytest` is the recommended way.
if __name__ == '__main__':
    # A simple way to run tests without installing pytest
    print("--- Running Model Tests ---")
    
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, X_TEST_PATH, Y_TEST_PATH]):
        print("One or more required files are missing. Cannot run tests.")
        print("Please run train script first.")
    else:
        pred = predictor()
        data = test_data()
        
        print("\n1. Testing Initialization...")
        test_predictor_initialization(pred)
        print("✓ Initialization test passed.")
        
        print("\n2. Testing Prediction Accuracy (on 100 samples)...")
        test_prediction_accuracy(pred, data)
        # No assert here for non-pytest run, just visual check
        
        print("\n3. Testing Edge Cases...")
        test_edge_case_zero_input(pred)
        test_edge_case_extreme_values(pred)
        
        print("\n4. Testing Invalid Input Shape...")
        try:
            test_invalid_input_shape(pred)
            print("✓ Invalid shape test passed.")
        except AssertionError:
            print("✗ Invalid shape test failed.")

        print("\n--- All tests finished ---")

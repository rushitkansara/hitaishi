
import tensorflow as tf
import numpy as np
import pickle
from typing import Tuple, Dict, List
import os

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

import tensorflow as tf
import numpy as np
import pickle
from typing import Tuple, Dict, List
import os

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

class HealthRiskPredictor:
    """
    Real-time health risk prediction engine using a Keras model.
    """

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initializes the predictor by loading the Keras model.
        """
        print(f"Initializing HealthRiskPredictor with Keras model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Class names aligned with V2 model (13 classes)
        self.class_names = [
            "Stable", "Monitor", "Heart_Attack", "Arrhythmia", "Heart_Failure",
            "Hypoglycemia", "Hyperglycemia_DKA", "Respiratory_Distress", "Sepsis",
            "Stroke", "Shock", "Hypertensive_Crisis", "Fall_Unconscious"
        ]
        print(f"Predictor initialized successfully with {len(self.class_names)} classes.")

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, str, Dict[str, float]]:
        """
        Predicts health risk from a 600-second vital sign sequence.
        """
        if sequence.shape != (600, 7):
            raise ValueError(f"Expected input shape (600, 7), but got {sequence.shape}")

        # Normalize the sequence
        seq_reshaped = sequence.reshape(-1, 7)
        seq_normalized = self.scaler.transform(seq_reshaped)
        
        # Reshape for the model (1, 600, 7)
        seq_input = seq_normalized.reshape(1, 600, 7).astype(np.float32)

        # Run model inference
        output = self.model.predict(seq_input)[0]

        # Parse results
        risk_level = int(np.argmax(output))
        confidence = float(output[risk_level])
        risk_name = self.class_names[risk_level]

        probabilities = {name: float(prob) for name, prob in zip(self.class_names, output)}

        return risk_level, confidence, risk_name, probabilities

    def predict_batch(self, sequences: np.ndarray) -> List[Tuple[int, float, str]]:
        """
        Performs batch prediction on multiple sequences.
        """
        predictions = []
        for seq in sequences:
            risk, conf, name, _ = self.predict(seq)
            predictions.append((risk, conf, name))
        return predictions

# Usage Example
if __name__ == '__main__':
    print("--- Inference Engine Usage Example ---")
    
    # Define paths
    MODEL_PATH = config.HEALTH_LSTM_MODEL_PATH
    SCALER_PATH = config.SCALER_PARAMS_PATH
    
    # Check if model and scaler exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or scaler file not found.")
        print("Please run the training script first.")
    else:
        # Initialize predictor
        predictor = HealthRiskPredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH
        )

        # Create a random test sequence
        print("\n--- Testing with a random sequence ---")
        test_seq = np.random.rand(60, 7) * 100 # Use a more realistic range
        
        try:
            risk, confidence, name, probs = predictor.predict(test_seq)
            
            print(f"Predicted Condition: {name}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Risk Level (index): {risk}")
            
            # Print top 3 probabilities
            print("\nTop 3 Probabilities:")
            top_3 = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
            for class_name, prob in top_3:
                print(f"  - {class_name}: {prob:.2%}")

        except Exception as e:
            print(f"An error occurred during prediction: {e}")

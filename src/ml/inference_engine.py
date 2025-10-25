
import tensorflow as tf
import numpy as np
import pickle
from typing import Tuple, Dict, List
import os

class HealthRiskPredictor:
    """
    Real-time health risk prediction engine using a Keras model.
    """

    def __init__(self, model_path: str, scaler_path: str):
        """
        Initializes the predictor by loading the Keras model and the scaler.
        """
        print(f"Initializing HealthRiskPredictor with model: {model_path}")
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Class names from the spec
        self.class_names = [
            'Stable', 'Monitor', 'Heart_Attack', 'Arrhythmia',
            'Heart_Failure', 'Hypoglycemia', 'Hyperglycemia_DKA',
            'Respiratory_Distress', 'Sepsis', 'Stroke',
            'Shock', 'Hypertensive_Crisis', 'Fall_Unconscious'
        ]
        print("Predictor initialized successfully.")

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, str, Dict[str, float]]:
        """
        Predicts health risk from a single 60-second vital sign sequence.

        Args:
            sequence: numpy array of shape (60, 8) with raw vital sign data.

        Returns:
            A tuple containing:
            - risk_level (int): The predicted class index (0-12).
            - confidence (float): The model's confidence in the prediction (0-1).
            - risk_name (str): The name of the predicted class.
            - probabilities (Dict[str, float]): A dictionary of all class probabilities.
        """
        if sequence.shape != (60, 8):
            raise ValueError(f"Expected input shape (60, 8), but got {sequence.shape}")

        # Normalize the sequence
        seq_reshaped = sequence.reshape(-1, 8)
        seq_normalized = self.scaler.transform(seq_reshaped)
        
        # Reshape for the model and ensure correct type
        seq_input = seq_normalized.reshape(1, 60, 8).astype(np.float32)

        # Run inference
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

        Args:
            sequences: A numpy array of sequences, shape (num_sequences, 60, 8).

        Returns:
            A list of tuples, where each tuple contains (risk_level, confidence, risk_name).
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
    MODEL_PATH = 'models/health_lstm_model.h5' # Use the unquantized model
    SCALER_PATH = 'data-gen/data/scaler_params.pkl'
    
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
        test_seq = np.random.rand(60, 8) * 100 # Use a more realistic range
        
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

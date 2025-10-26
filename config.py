import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory for the project

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Generation Paths
DATA_GEN_DIR = os.path.join(BASE_DIR, 'data-gen')
DATA_DIR = os.path.join(DATA_GEN_DIR, 'data')

TRAINING_SEQUENCES_PATH = os.path.join(DATA_DIR, 'training_sequences.npz')
TRAINING_LABELS_PATH = os.path.join(DATA_DIR, 'training_labels.npy')
AUGMENTED_SEQUENCES_PATH = os.path.join(DATA_DIR, 'augmented_sequences.npz')
AUGMENTED_LABELS_PATH = os.path.join(DATA_DIR, 'augmented_labels.npy')
PATIENT_PROFILES_PATH = os.path.join(DATA_DIR, 'patient_profiles.json')
SCALER_PARAMS_PATH = os.path.join(DATA_DIR, 'scaler_params.pkl')
GENERATION_REPORT_PATH = os.path.join(DATA_DIR, 'generation_report.txt')
SAMPLE_VISUALIZATIONS_PATH = os.path.join(DATA_DIR, 'sample_visualizations.png')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.npy')

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
HEALTH_LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'health_lstm_model.h5')
HEALTH_MODEL_QUANTIZED_PATH = os.path.join(MODELS_DIR, 'health_model_quantized.tflite')
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, 'confusion_matrix.png')
EVALUATION_REPORT_PATH = os.path.join(MODELS_DIR, 'evaluation_report.txt')
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.json')
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# Log Paths
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
TENSORBOARD_LOG_DIR = os.path.join(LOGS_DIR, 'tensorboard')

# Backend Configuration
BACKEND_HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 8000))
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

# Twilio Configuration (read from environment variables)
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
RECIPIENT_PHONE_NUMBER = os.environ.get("RECIPIENT_PHONE_NUMBER")

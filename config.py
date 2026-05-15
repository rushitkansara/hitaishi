import os
from dotenv import load_dotenv

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file explicitly
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_PATH, override=True)

# Data Generation Paths
DATA_GEN_DIR = os.path.join(BASE_DIR, 'data_gen')
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

# Procedural Data Paths
DATA_PROCEDURAL_X_TRAIN_PATH = os.path.join(DATA_DIR, 'procedural_X_train.npz')
DATA_PROCEDURAL_Y_TRAIN_PATH = os.path.join(DATA_DIR, 'procedural_y_train.npy')
DATA_PROCEDURAL_X_TEST_PATH = os.path.join(DATA_DIR, 'procedural_X_test.npz')
DATA_PROCEDURAL_Y_TEST_PATH = os.path.join(DATA_DIR, 'procedural_y_test.npy')
DATA_PROCEDURAL_SCALER_PATH = os.path.join(DATA_DIR, 'procedural_scaler.pkl')

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
MODEL_V1_2_H5_PATH = os.path.join(BASE_DIR, 'experiments', 'v2_1_hybrid_20260417_105336', 'best_model.keras')
MODEL_V1_2_TFLITE_PATH = os.path.join(MODELS_DIR, 'health_model_quantized.tflite')
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
TWILIO_MESSAGING_SERVICE_SID = os.environ.get("TWILIO_MESSAGING_SERVICE_SID")
RECIPIENT_PHONE_NUMBER = os.environ.get("RECIPIENT_PHONE_NUMBER")

# TextBelt Configuration
TEXTBELT_URL = os.environ.get("TEXTBELT_URL", "https://textbelt.com/text")
TEXTBELT_KEY = os.environ.get("TEXTBELT_KEY", "textbelt") # 'textbelt' is the default key for free tier
# ClickSend Configuration
CLICKSEND_USERNAME = os.environ.get("CLICKSEND_USERNAME")
CLICKSEND_API_KEY = os.environ.get("CLICKSEND_API_KEY")
CLICKSEND_SOURCE = os.environ.get("CLICKSEND_SOURCE", "Hitaishi")

SMS_PROVIDER = os.environ.get("SMS_PROVIDER", "email_gateway") # 'twilio', 'textbelt', 'email_gateway', 'local_sendmail', or 'clicksend'

# SMTP Configuration (for email_gateway provider)
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_FROM = os.environ.get("SMTP_FROM")

print(f"DEBUG: SMS Config - Provider: {SMS_PROVIDER}, Twilio Phone: {TWILIO_PHONE_NUMBER}")

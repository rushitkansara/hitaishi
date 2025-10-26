import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Add the ml directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'ml')))
from inference_engine import HealthRiskPredictor
from .twilio_service import send_alert

# Import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

app = FastAPI()

# CORS Middleware
origins = [
    config.FRONTEND_URL,
    "http://localhost:3000", # For local development
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler
MODEL_PATH = config.HEALTH_MODEL_QUANTIZED_PATH
SCALER_PATH = config.SCALER_PARAMS_PATH
predictor = HealthRiskPredictor(MODEL_PATH, SCALER_PATH)

class Sequence(BaseModel):
    patient_name: str
    sequence: list

class Recipient(BaseModel):
    number: str

@app.get("/")
def read_root():
    return {"message": "Hitaishi Backend is running"}

@app.post("/predict")
def predict(data: Sequence):
    sequence = np.array(data.sequence)
    risk_level, confidence, risk_name, probabilities = predictor.predict(sequence)

    if risk_name not in ['Stable', 'Monitor']:
        send_alert(data.patient_name, risk_name)

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "risk_name": risk_name,
        "probabilities": probabilities
    }

@app.post("/set_recipient")
def set_recipient(data: Recipient):
    with open("recipient_number.txt", "w") as f:
        f.write(data.number)
    return {"message": "Recipient number saved successfully."}

@app.get("/get_recipient")
def get_recipient():
    try:
        with open("recipient_number.txt", "r") as f:
            number = f.read()
        return {"number": number}
    except FileNotFoundError:
        return {"number": ""}

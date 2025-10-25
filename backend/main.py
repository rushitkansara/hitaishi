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

app = FastAPI()

# CORS Middleware
origins = [
    "http://localhost:3000",
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
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'health_lstm_model.h5'))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-gen', 'data', 'scaler_params.pkl'))
predictor = HealthRiskPredictor(MODEL_PATH, SCALER_PATH)

class Sequence(BaseModel):
    patient_name: str
    sequence: list

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

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
from src.ml.sim_gen import generate_patient_specific_stable_sequence, generate_patient_specific_emergency_sequence

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
SCALER_PATH = config.PROCEDURAL_SCALER_PATH
predictor = HealthRiskPredictor(MODEL_PATH, SCALER_PATH)

class PatientProfile(BaseModel):
    name: str
    age: int
    gender: str
    activity_level: str
    primary_condition: str

class SimulationRequest(BaseModel):
    patient_profile: PatientProfile
    emergency_type: int
    emergency_contact: str

@app.get("/")
def read_root():
    return {"message": "Hitaishi Backend is running"}

@app.post("/generate_stable_data")
def generate_stable_data(profile: PatientProfile):
    profile_factors = {
        "name": profile.name,
        "age": profile.age,
        "gender": profile.gender,
        "activity_level": profile.activity_level,
        "primary_condition": profile.primary_condition
    }
    sequence = generate_patient_specific_stable_sequence(profile_factors)
    return {"sequence": sequence.tolist()}

@app.post("/simulate_emergency")
def simulate_emergency(request: SimulationRequest):
    profile_factors = {
        "name": request.patient_profile.name,
        "age": request.patient_profile.age,
        "gender": request.patient_profile.gender,
        "activity_level": request.patient_profile.activity_level,
        "primary_condition": request.patient_profile.primary_condition
    }
    emergency_type = request.emergency_type
    emergency_contact = request.emergency_contact

    # Generate emergency sequence
    sequence = generate_patient_specific_emergency_sequence(profile_factors, emergency_type)

    # Run inference
    risk_level, confidence, risk_name, probabilities = predictor.predict(sequence)

    alert_status = "Not Sent"
    if risk_name not in ['Stable', 'Monitor']:
        sid = send_alert(request.patient_profile.name, risk_name, emergency_contact)
        if sid:
            alert_status = "Sent"
        else:
            alert_status = "Failed"

    return {
        "sequence": sequence.tolist(),
        "prediction": {
            "risk_level": risk_level,
            "confidence": confidence,
            "risk_name": risk_name,
            "probabilities": probabilities
        },
        "alert_status": alert_status
    }

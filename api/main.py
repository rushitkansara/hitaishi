from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Optional, Dict
import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path to import config and modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'ml')))

logger.info("Starting Backend Initialization")

try:
    from inference_engine import HealthRiskPredictor
    import config
    from src.ml.sim_gen import generate_patient_specific_stable_sequence, generate_patient_specific_emergency_sequence
    from src.ml.simulation_manager import get_sim_manager
    from api.database import get_db, Patient, EmergencyContact, init_db
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import SQLAlchemyError
    logger.info("Modules imported successfully")
except Exception as e:
    logger.exception("Failed to import modules")
    sys.exit(1)

app = FastAPI()

# CORS Middleware
origins = [
    config.FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "https://hitaishi.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.ml.model_registry import get_predictor

# Access predictor where needed using: predictor = get_predictor()
class Contact(BaseModel):
    name: str
    phone: str
    verified: int = 0 # Default to not verified

class PatientProfile(BaseModel):
    name: str
    age: int
    gender: str
    activity_level: str
    primary_condition: str
    contacts: List[Contact]

class SimulationRequest(BaseModel):
    patient_profile: PatientProfile
    emergency_type: int

@app.get("/api/v1_2/patients")
async def get_all_patients(db: Session = Depends(get_db)):
    try:
        patients_db = db.query(Patient).all()
        # Convert to dict for JSON response, including related contacts
        patients_list = []
        for p in patients_db:
            patient_dict = {
                "id": p.id,
                "name": p.name,
                "age": p.age,
                "gender": p.gender,
                "activity_level": p.activity_level,
                "primary_condition": p.primary_condition,
                "created_at": p.created_at,
                "contacts": [c.__dict__ for c in p.contacts] if p.contacts else []
            }
            patients_list.append(patient_dict)
        return patients_list
    except SQLAlchemyError as e:
        print(f"Error fetching patients: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patient list.")

@app.get("/api/v1_2/patients/{patient_id}")
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    try:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_dict = {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "activity_level": p.activity_level,
            "primary_condition": p.primary_condition,
            "created_at": p.created_at,
            "contacts": [c.__dict__ for c in p.contacts] if p.contacts else []
        }
        return patient_dict
    except SQLAlchemyError as e:
        print(f"Error fetching patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patient {patient_id}.")

# --- Simulation Manager Endpoints ---

@app.post("/api/v1_2/patients")
async def init_simulation(profile: PatientProfile, db: Session = Depends(get_db)):
    """Initializes a live simulation buffer for a patient and saves to DB."""
    print(f"DEBUG: Received request to /api/v1_2/patients for {profile.name}")
    try:
        # Check if patient already exists by name
        existing_patient = db.query(Patient).filter(Patient.name == profile.name).first()
        manager = get_sim_manager()
        
        if existing_patient:
            print(f"Patient {profile.name} already exists with ID {existing_patient.id}.")
            # Initialize simulation if not present in memory
            if str(existing_patient.id) not in manager.active_sessions:
                conditions = [existing_patient.primary_condition] if existing_patient.primary_condition != "Healthy" else []
                manager.initialize_patient(
                    str(existing_patient.id), 
                    existing_patient.name,
                    int(existing_patient.age), 
                    existing_patient.gender.lower(), 
                    conditions, 
                    existing_patient.activity_level,
                    contacts=[{"name": c.name, "phone": c.phone, "verified": c.verified} for c in existing_patient.contacts]
                )
            return {"status": "exists", "message": f"Patient {profile.name} already exists.", "patient_id": existing_patient.id}

        # Create patient in DB
        new_patient = Patient(
            name=profile.name,
            age=int(profile.age),
            gender=profile.gender.lower(),
            activity_level=profile.activity_level,
            primary_condition=profile.primary_condition
        )
        db.add(new_patient)
        db.flush() # Flush to get the new patient's ID

        # Create emergency contacts in DB
        for contact_data in profile.contacts:
            if contact_data.name and contact_data.phone:
                new_contact = EmergencyContact(
                    patient_id=new_patient.id,
                    name=contact_data.name,
                    phone=contact_data.phone,
                    verified=contact_data.verified
                )
                db.add(new_contact)
        
        db.commit()
        db.refresh(new_patient)

        # Initialize simulation buffer using numeric ID as key
        conditions = [profile.primary_condition] if profile.primary_condition != "Healthy" else []
        manager.initialize_patient(
            str(new_patient.id), 
            profile.name,
            int(profile.age), 
            profile.gender.lower(), 
            conditions, 
            profile.activity_level,
            contacts=[c.dict() for c in profile.contacts]
        )
        
        return {"status": "success", "message": f"Simulation initialized for {profile.name}", "patient_id": new_patient.id}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"DEBUG: Database error during patient initialization: {e}")
        raise HTTPException(status_code=500, detail="Database error during patient initialization.")
    except Exception as e:
        db.rollback()
        print(f"DEBUG: init_simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from api.verification_service import send_verification_sms, verify_contact_code

@app.post("/api/v1_2/contacts/{contact_id}/verify/send")
def send_contact_verification(contact_id: str, patient_name: str, phone: str):
    print(f"DEBUG: VERIFY SEND HIT for {contact_id}, phone {phone}")
    return send_verification_sms(contact_id, phone, patient_name)

@app.post("/api/v1_2/contacts/{contact_id}/verify/confirm")
def confirm_contact_verification(contact_id: str, payload: dict):
    return verify_contact_code(contact_id, payload.get("code", ""))

import traceback
@app.post("/api/v1_2/sim/trigger/{state_code}")
def trigger_sim_emergency(patient_id: str, state_code: int):
    """Triggers an emergency in the live simulation."""
    try:
        manager = get_sim_manager()
        if patient_id not in manager.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session for patient {patient_id} not found. Please initialize patient.")
        manager.trigger_emergency(patient_id, state_code)
        return {"status": "success"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1_2/sim/abort/{patient_id}")
def abort_sim_emergency(patient_id: str):
    """Aborts an emergency simulation and returns to stable."""
    try:
        manager = get_sim_manager()
        manager.abort_simulation(patient_id)
        return {"status": "success"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1_2/patients/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Deletes a patient from the database and removes their simulation session."""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        db.delete(patient)
        db.commit()
        
        # Also remove from SimulationManager active sessions using numeric ID
        manager = get_sim_manager()
        if str(patient_id) in manager.active_sessions:
            del manager.active_sessions[str(patient_id)]
        
        return {"status": "success", "message": f"Patient deleted successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error deleting patient: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete patient from database.")
    except Exception as e:
        print(f"Error during patient deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1_2/sim/tick/{patient_id}")
def get_sim_tick(patient_id: str):
    """Poll endpoint for 1s heartbeat, inference and rationale."""
    try:
        manager = get_sim_manager()
        return manager.get_next_tick(patient_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)

import random
import time
from datetime import datetime
import config
from .sms_service import send_sms

# In-memory store for pending verifications
# { contact_id: {"code": str, "expires_at": float, "attempts": int} }
PENDING_VERIFICATIONS = {}

HERO_CODES = [
    "ironman", "superman", "batman", "spiderman", "hulk",
    "blackpanther", "thor", "wonderwoman", "captainamerica", "flash"
]

def generate_hero_code() -> str:
    return random.choice(HERO_CODES)

def send_verification_sms(contact_id: str, phone: str, patient_name: str):
    code = generate_hero_code()
    expires_at = time.time() + 600 # 10 min window
    
    print(f"DEBUG: Generating verification for {contact_id}. Phone: {phone}, Code: {code}")
    
    PENDING_VERIFICATIONS[contact_id] = {
        "code": code,
        "expires_at": expires_at,
        "attempts": 0
    }
    
    body = f"Hitaishi: You are being added as an emergency contact for {patient_name}. Your verification word is: {code.upper()}"
    result = send_sms(phone, body)
    
    if result["status"] == "sent":
        return {"status": "sent", "expires_in": 600, "code": code}
    else:
        return {"status": "failed", "error": result["error"]}

def verify_contact_code(contact_id: str, submitted: str):
    print(f"DEBUG: Verifying code for {contact_id}. Submitted: '{submitted}'")
    
    # Support for simple Yes/No UI flow
    if submitted == 'FORCE_VERIFY':
        print("DEBUG: Force verifying via UI confirmation")
        if contact_id in PENDING_VERIFICATIONS:
            del PENDING_VERIFICATIONS[contact_id]
        return {"valid": True}

    pending = PENDING_VERIFICATIONS.get(contact_id)
    
    if not pending:
        print(f"DEBUG: No pending verification found for {contact_id}")
        return {"valid": False, "reason": "No verification pending"}
    
    print(f"DEBUG: Pending data: {pending}")
    
    if time.time() > pending["expires_at"]:
        print("DEBUG: Code expired")
        del PENDING_VERIFICATIONS[contact_id]
        return {"valid": False, "reason": "Code expired, resend"}
    if pending["attempts"] >= 3:
        print("DEBUG: Too many attempts")
        return {"valid": False, "reason": "Too many attempts, resend"}
    
    pending["attempts"] += 1
    
    if submitted.strip().lower() != pending["code"].lower():
        print(f"DEBUG: Code mismatch. Expected {pending['code'].lower()}, got {submitted.strip().lower()}")
        return {"valid": False, "reason": "Wrong word"}
    
    print("DEBUG: Verification successful")
    del PENDING_VERIFICATIONS[contact_id]
    return {"valid": True}

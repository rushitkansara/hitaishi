import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from .sms_service import send_sms

def send_alert(patient_name: str, condition: str, recipient_phone_number: str):
    if not recipient_phone_number:
        print("Recipient phone number not provided. Cannot send SMS alert.")
        return None

    body = f"ALERT: Patient {patient_name} is in a critical condition: {condition}"
    result = send_sms(recipient_phone_number, body)
    
    if result["status"] == "sent":
        return result["id"]
    return None

import os
from twilio.rest import Client

# Import configuration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

def send_alert(patient_name: str, condition: str, recipient_phone_number: str):
    if not recipient_phone_number:
        print("Recipient phone number not provided. Cannot send SMS alert.")
        return None

    try:
        message = client.messages.create(
            to=recipient_phone_number,
            from_=config.TWILIO_PHONE_NUMBER,
            body=f"ALERT: Patient {patient_name} is in a critical condition: {condition}"
        )
        print(f"SMS alert sent successfully: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")
        return None

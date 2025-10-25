import os
from twilio.rest import Client

# Your Account SID and Auth Token from twilio.com/console
# TODO: Replace with your actual credentials
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "ACb199635168604a56cf6e64dace301f3e")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "dc8ec2ec449143a0fb0de1c355b0b856")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+18314981738")
RECIPIENT_PHONE_NUMBER = os.environ.get("RECIPIENT_PHONE_NUMBER", "+918732920909")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_alert(patient_name: str, condition: str):
    try:
        message = client.messages.create(
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            body=f"ALERT: Patient {patient_name} is in a critical condition: {condition}"
        )
        print(f"SMS alert sent successfully: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")
        return None

import time
import requests
import config
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from .email_gateways import PROVIDERS

def send_sms(to_phone: str, message_body: str):
    """
    Unified SMS sender that chooses between Twilio, TextBelt, Email Gateway, and Local Sendmail.
    """
    import os
    print("--- SMS DEBUG START ---")
    print(f"ENV SMS_PROVIDER: {os.environ.get('SMS_PROVIDER')}")
    print(f"CONFIG SMS_PROVIDER: {config.SMS_PROVIDER}")
    print("--- SMS DEBUG END ---")

    if config.SMS_PROVIDER == "textbelt":
        return _send_via_textbelt(to_phone, message_body)
    elif config.SMS_PROVIDER == "email_gateway":
        return _send_via_email_gateway(to_phone, message_body)
    elif config.SMS_PROVIDER == "local_sendmail":
        return _send_via_local_sendmail(to_phone, message_body)
    else:
        return _send_via_twilio(to_phone, message_body)
def _send_via_local_sendmail(to_phone: str, message_body: str):
    """
    Sends SMS using the local system 'sendmail' command to carrier gateways.
    No credentials required.
    """
    print(f"DEBUG: Sending SMS via Local Sendmail to {to_phone}")
    phone_digits = ''.join(filter(str.isdigit, to_phone))
    success_count = 0
    
    for category in ['us', 'canada', 'intl']:
        for provider_template in PROVIDERS[category]:
            recipient = provider_template % phone_digits
            try:
                # Construct the email message
                msg = f"To: {recipient}\nSubject: Hitaishi Alert\n\n{message_body}"
                # Pipe it to sendmail
                process = subprocess.Popen(['sendmail', '-t'], stdin=subprocess.PIPE)
                process.communicate(input=msg.encode())
                success_count += 1
            except Exception as e:
                print(f"DEBUG: Sendmail failed for {recipient}: {e}")
    
    if success_count > 0:
        print(f"DEBUG: Local Sendmail broadcasted to {success_count} providers")
        return {"status": "sent", "id": "sendmail-broadcast"}
    else:
        return {"status": "failed", "error": "Local sendmail failed to broadcast"}

def _send_via_email_gateway(to_phone: str, message_body: str):
    """
    Sends SMS by emailing carrier gateways (Free TextBelt method).
    """
    print(f"DEBUG: Sending SMS via Email Gateway to {to_phone}")
    if not all([config.SMTP_HOST, config.SMTP_USER, config.SMTP_PASS]):
        return {"status": "failed", "error": "SMTP configuration missing in .env"}

    # Clean phone number (digits only)
    phone_digits = ''.join(filter(str.isdigit, to_phone))
    
    success_count = 0
    errors = []

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            
            # Try ALL provider categories for maximum coverage
            for category in ['us', 'canada', 'intl']:
                for provider_template in PROVIDERS[category]:
                    recipient = provider_template % phone_digits
                    msg = MIMEText(message_body)
                    msg['From'] = config.SMTP_FROM or config.SMTP_USER
                    msg['To'] = recipient
                    msg['Subject'] = ""
                    
                    try:
                        server.send_message(msg)
                        success_count += 1
                    except Exception as e:
                        errors.append(f"{recipient}: {e}")
        
        if success_count > 0:
            print(f"DEBUG: Email Gateway broadcasted to {success_count} providers")
            return {"status": "sent", "id": "gateway-broadcast"}
        else:
            print(f"DEBUG: Email Gateway Failed. Errors: {errors[:5]}")
            return {"status": "failed", "error": f"Failed to send via any gateway. First error: {errors[0] if errors else 'Unknown'}"}
            
    except Exception as e:
        print(f"DEBUG: SMTP Error: {e}")
        return {"status": "failed", "error": str(e)}

def _send_via_textbelt(to_phone: str, message_body: str):
    print(f"DEBUG: Sending SMS via TextBelt to {to_phone}")
    try:
        data = {
            'number': to_phone,
            'message': message_body,
        }
        if config.TEXTBELT_KEY:
            data['key'] = config.TEXTBELT_KEY

        response = requests.post(config.TEXTBELT_URL, data=data)
        
        print(f"DEBUG: TextBelt Response Status: {response.status_code}")
        
        try:
            result = response.json()
        except Exception:
            print(f"DEBUG: TextBelt Raw Response: {response.text}")
            return {"status": "failed", "error": f"Invalid JSON response: {response.text[:100]}"}

        if result.get('success'):
            print(f"DEBUG: TextBelt SMS sent successfully. ID: {result.get('textId')}")
            return {"status": "sent", "id": result.get('textId')}
        else:
            error_msg = result.get('message') or result.get('error') or f"Unknown TextBelt error (Success: False, Status: {response.status_code})"
            print(f"DEBUG: TextBelt SMS Failed: {error_msg}")
            return {"status": "failed", "error": error_msg}
    except Exception as e:
        print(f"DEBUG: TextBelt Exception: {e}")
        return {"status": "failed", "error": str(e)}

def _send_via_twilio(to_phone: str, message_body: str):
    print(f"DEBUG: Sending SMS via Twilio to {to_phone}")
    try:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        
        message_params = {
            "to": to_phone,
            "body": message_body
        }

        if config.TWILIO_MESSAGING_SERVICE_SID:
            message_params["messaging_service_sid"] = config.TWILIO_MESSAGING_SERVICE_SID
        else:
            message_params["from_"] = config.TWILIO_PHONE_NUMBER

        message = client.messages.create(**message_params)
        print(f"DEBUG: Twilio SMS sent successfully. SID: {message.sid}")
        return {"status": "sent", "id": message.sid}
    except Exception as e:
        error_msg = str(e)
        if "is not a Twilio phone number" in error_msg:
            error_msg = f"Twilio Error: The number {config.TWILIO_PHONE_NUMBER} is not recognized by your account."
        print(f"DEBUG: Twilio SMS Error: {error_msg}")
        return {"status": "failed", "error": error_msg}

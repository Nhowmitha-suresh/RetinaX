import os
from io import BytesIO

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[!] Twilio not installed - WhatsApp notifications disabled")

def send_whatsapp_notification(patient_name: str, severity_label: str, confidence: float, pdf_filename: str = None):
    """Send WhatsApp notification with PDF report link after DR report generation"""
    if not TWILIO_AVAILABLE:
        print("[!] Twilio not available")
        return False
    try:
        account_sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
        auth_token    = os.getenv("TWILIO_AUTH_TOKEN", "")
        from_whatsapp = os.getenv("TWILIO_WHATSAPP", "whatsapp:+14155238886")
        to_whatsapp   = os.getenv("RECIPIENT_PHONE", "")
        public_host   = os.getenv("PUBLIC_HOST", "http://localhost:8000")

        if not account_sid or not auth_token or not to_whatsapp:
            print("[!] Twilio credentials incomplete in .env. Skipping WhatsApp notification.")
            return False

        if not to_whatsapp.startswith("whatsapp:"):
            to_whatsapp = f"whatsapp:{to_whatsapp}"

        client = TwilioClient(account_sid, auth_token)

        message_body = (
            f"*RetinaX - DR Screening Result*\n\n"
            f"Patient: {patient_name}\n"
            f"Diagnosis: {severity_label}\n"
            f"Confidence: {confidence:.1f}%\n\n"
            f"Your full medical report is generated.\n"
            f"Please consult your ophthalmologist for further evaluation.\n"
            f"- Team Thiran | RetinaX AI"
        )

        media_url = None
        if pdf_filename and public_host and not public_host.startswith("http://localhost") and not public_host.startswith("http://127.0.0.1"):
            media_url = f"{public_host.rstrip('/')}/reports/{pdf_filename}"

        if media_url:
            message = client.messages.create(body=message_body, from_=from_whatsapp, to=to_whatsapp, media_url=[media_url])
        else:
            message = client.messages.create(body=message_body, from_=from_whatsapp, to=to_whatsapp)

        print(f"[+] WhatsApp sent with PDF! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"[!] WhatsApp notification failed: {e}")
        return False


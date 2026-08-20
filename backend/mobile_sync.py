import os
import time
import uuid
import socket
import secrets
import io
import base64
from typing import Dict, Any, Optional

# In-memory store for active mobile capture sessions
# Format: { session_id: { "session_id": str, "token": str, "status": str, "created_at": float, "expires_at": float, ... } }
active_sessions: Dict[str, Dict[str, Any]] = {}

def get_local_ip() -> str:
    """Retrieve the primary local IPv4 address of the host machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"

def get_mobile_base_url() -> str:
    """
    Resolve the base URL for mobile QR code connections.
    Uses MOBILE_BASE_URL from .env if present; otherwise defaults to http://<LAN_IP>:8000.
    Never uses hardcoded localhost or 127.0.0.1 for mobile QR code URLs.
    """
    env_url = os.getenv("MOBILE_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    
    lan_ip = get_local_ip()
    return f"http://{lan_ip}:8000"

def generate_qr_base64(payload: str) -> Optional[str]:
    """Generate a base64 encoded PNG QR code image for a given payload string."""
    try:
        import qrcode
        img = qrcode.make(payload)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[!] Server-side QR code module info: {e}")
        return None

def create_session(expiry_seconds: int = 300) -> Dict[str, Any]:
    """
    Create a new secure mobile capture session with unique session_id, cryptographically
    secure random token, and expiration timestamp.
    """
    session_id = str(uuid.uuid4())[:8]
    token = secrets.token_hex(16)
    created_at = time.time()
    expires_at = created_at + expiry_seconds
    base_url = get_mobile_base_url()

    mobile_url = f"{base_url}/mobile-camera?session={session_id}&token={token}"
    local_url = f"http://localhost:8000/mobile-camera?session={session_id}&token={token}"
    qr_image_base64 = generate_qr_base64(mobile_url)

    session_data = {
        "session_id": session_id,
        "token": token,
        "status": "waiting",  # waiting -> image_received -> processing -> pass / fail
        "created_at": created_at,
        "expires_at": expires_at,
        "image_bytes": None,
        "image_b64": None,
        "quality": None,
        "fundus_validation": None,
        "prediction": None,
        "reason": None,
        "mobile_url": mobile_url,
        "local_url": local_url,
        "qr_image_base64": qr_image_base64
    }

    active_sessions[session_id] = session_data
    cleanup_old_sessions()

    print(f"[+] Created Mobile Session {session_id} -> Mobile LAN URL: {mobile_url}")

    return {
        "session_id": session_id,
        "token": token,
        "expires_at": expires_at,
        "expires_in_seconds": expiry_seconds,
        "mobile_url": mobile_url,
        "local_url": local_url,
        "qr_payload": mobile_url,
        "qr_image_base64": qr_image_base64
    }

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session data by session_id if not expired."""
    session = active_sessions.get(session_id)
    if not session:
        return None

    # Expiry Check
    if time.time() > session["expires_at"]:
        session["status"] = "expired"
        return session

    return session

def validate_session_token(session_id: str, token: str) -> bool:
    """Validate that the session exists, matches the secure token, and is active."""
    session = get_session(session_id)
    if not session:
        return False
    if session.get("status") == "expired":
        return False
    return session.get("token") == token

def update_session(session_id: str, **kwargs):
    """Update fields in an active session."""
    if session_id in active_sessions:
        active_sessions[session_id].update(kwargs)

def cleanup_old_sessions(max_age_seconds: int = 1800):
    """Remove sessions older than max_age_seconds (30 minutes)."""
    now = time.time()
    to_delete = [sid for sid, data in active_sessions.items() if now - data["created_at"] > max_age_seconds]
    for sid in to_delete:
        del active_sessions[sid]

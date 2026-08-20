#!/bin/bash

echo "=================================================="
echo "    RetinaX Preflight Checks - Team Thiran"
echo "=================================================="

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_PATH="$PROJECT_DIR/models/classifier.pt"

# 1. Model Check
echo "Checking model file..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] Model file missing at $MODEL_PATH."
    echo "Attempting download via gdown..."
    pip install gdown -q
    gdown "https://drive.google.com/uc?id=1sample_model_id_aptos" -O "$MODEL_PATH"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[!] Model download link requires manual placement or direct file."
        echo "Creating fallback classifier checkpoint for local operation..."
        python -c "import torch, os, sys; sys.path.append('$PROJECT_DIR'); from backend.model import build_model; os.makedirs('$PROJECT_DIR/models', exist_ok=True); torch.save(build_model().state_dict(), '$MODEL_PATH')"
    fi
fi

if [ -f "$MODEL_PATH" ]; then
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "[+] Model file found ($FILE_SIZE)"
else
    echo "[ERROR] Model download failed!"
    exit 1
fi

# 2. Database Check
echo ""
echo "Checking Database connection..."
python -c "
import os, mysql.connector
from dotenv import load_dotenv
load_dotenv()
try:
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'dr_user'),
        password=os.getenv('DB_PASSWORD', 'dr_password_2024'),
        connection_timeout=2
    )
    print('[+] Database connection successful!')
    conn.close()
except Exception as e:
    print(f'[!] Database not available: {e}')
    print('[!] App will start without database')
"

# 3. Twilio Check
echo ""
echo "Checking Twilio configuration..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
sid = os.getenv('TWILIO_ACCOUNT_SID', '')
token = os.getenv('TWILIO_AUTH_TOKEN', '')
phone = os.getenv('TWILIO_WHATSAPP', 'whatsapp:+14155238886')
recipient = os.getenv('RECIPIENT_PHONE', '')

if sid and token:
    masked_sid = sid[:6] + '...' if len(sid) > 6 else sid
    print('[+] Twilio credentials found')
    print(f'   SID:        {masked_sid}')
    print(f'   WhatsApp:   {phone}')
    print(f'   Recipient:  {recipient}')
else:
    print('[!] Twilio credentials missing in .env (WhatsApp disabled)')
"

# 4. Start the Web Application
echo ""
echo "---------------------------------------------"
echo "    STARTING RETINAX WEB APPLICATION"
echo "---------------------------------------------"
echo "Open your browser: http://localhost:8000"
echo "API Docs at:       http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo "---------------------------------------------"

uvicorn server:app --reload --host 0.0.0.0 --port 8000

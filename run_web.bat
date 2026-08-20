@echo off
echo ==================================================
echo     RetinaX Preflight Checks - Team Thiran
echo ==================================================

set MODEL_PATH=models\classifier.pt

echo Checking model file...
if not exist "%MODEL_PATH%" (
    echo [!] Model file missing at %MODEL_PATH%. Initializing fallback model weights...
    python -c "import torch, os, sys; sys.path.append('.'); from backend.model import build_model; os.makedirs('models', exist_ok=True); torch.save(build_model().state_dict(), '%MODEL_PATH%')"
)

if exist "%MODEL_PATH%" (
    echo [+] Model file found
) else (
    echo [ERROR] Model file initialization failed!
    exit /b 1
)

echo.
echo Checking Database connection...
python -c "import os, mysql.connector; from dotenv import load_dotenv; load_dotenv(); print('[+] Database connection successful!') if mysql.connector.connect(host=os.getenv('DB_HOST', 'localhost'), user=os.getenv('DB_USER', 'dr_user'), password=os.getenv('DB_PASSWORD', 'dr_password_2024'), connection_timeout=2) else None" 2>NUL || echo [!] Database not available, starting app without DB

echo.
echo Checking Twilio configuration...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); sid=os.getenv('TWILIO_ACCOUNT_SID',''); print('[+] Twilio credentials found') if sid else print('[!] Twilio credentials missing in .env (WhatsApp disabled)')"

echo.
echo ---------------------------------------------
echo     STARTING RETINAX WEB APPLICATION
echo ---------------------------------------------
echo Open your browser: http://localhost:8000
echo API Docs at:       http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo ---------------------------------------------

uvicorn server:app --reload --host 0.0.0.0 --port 8000

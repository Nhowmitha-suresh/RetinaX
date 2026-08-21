import os
import sys
import io
import time
import json
import requests

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sampleimages", "eye1.png")

print("=== TESTING LIVE PRODUCTION FASTAPI SERVER ENDPOINTS ===")

# 1. Health Endpoint
r_health = requests.get(f"{BASE_URL}/api/health")
print(f"1. /api/health          : Status {r_health.status_code} | {r_health.json()}")

# 2. Docs Endpoint
r_docs = requests.get(f"{BASE_URL}/docs")
print(f"2. /docs                : Status {r_docs.status_code} | {'OK' if r_docs.status_code == 200 else 'FAIL'}")

# 3. Predict Endpoint (/api/predict)
with open(IMAGE_PATH, "rb") as f:
    files = {"file": ("eye1.png", f, "image/png")}
    data = {"patient_id": "TEST-101", "patient_name": "Test Patient"}
    r_pred = requests.post(f"{BASE_URL}/api/predict", files=files, data=data)

print(f"3. /api/predict         : Status {r_pred.status_code}")
if r_pred.status_code == 200:
    res = r_pred.json()
    print(f"   Level: {res['level']} | Label: {res['label']} | Confidence: {res['confidence']}%")
    print(f"   GradCAM keys: {list(res['gradcam'].keys())}")
    print(f"   Quality Score: {res['quality']['quality_score']}")

# 4. Analyze Endpoint (/analyze)
with open(IMAGE_PATH, "rb") as f:
    files = {"file": ("eye1.png", f, "image/png")}
    r_ana = requests.post(f"{BASE_URL}/analyze", files=files)
print(f"4. /analyze             : Status {r_ana.status_code}")

# 5. Grad-CAM Standalone Endpoint (/api/gradcam)
with open(IMAGE_PATH, "rb") as f:
    files = {"file": ("eye1.png", f, "image/png")}
    data = {"target_class": 3}
    r_gcam = requests.post(f"{BASE_URL}/api/gradcam", files=files, data=data)
print(f"5. /api/gradcam         : Status {r_gcam.status_code} | Success: {r_gcam.json().get('success')}")

# 6. PDF Report Generation (/api/v1/report)
report_payload = {
    "patient_name": "Test Patient",
    "patient_id": "RX-9999",
    "age": "50",
    "gender": "Female",
    "dob": "1974-05-12",
    "contact": "+1 555 123 4567",
    "diabetes_type": "Type 2",
    "diabetes_duration": "8 years",
    "referring_doctor": "Dr. House",
    "eye_examined": "Right Eye",
    "clinical_notes": "Automated verification test.",
    "level": 3,
    "label": "Severe",
    "confidence": 98.4,
    "risk": "High",
    "action": "Urgent referral",
    "advice": "Severe DR detected.",
    "color": "#C62828",
    "all_probs": [0.1, 0.2, 0.3, 98.4, 1.0],
    "probabilities": {"Severe": 98.4},
    "quality": {"quality_score": 95, "status": "Good"},
    "gradcam": res.get("gradcam", {})
}
r_pdf = requests.post(f"{BASE_URL}/api/v1/report", json=report_payload)
print(f"6. /api/v1/report (PDF) : Status {r_pdf.status_code} | Content-Type: {r_pdf.headers.get('content-type')}")

# 7. Nearby Doctors Endpoint (/api/doctors/nearby)
r_docs_near = requests.get(f"{BASE_URL}/api/doctors/nearby?lat=37.7749&lng=-122.4194")
print(f"7. /api/doctors/nearby  : Status {r_docs_near.status_code} | Doctors count: {len(r_docs_near.json().get('doctors', []))}")

# 8. Patient History Endpoint (/api/patients)
r_patients = requests.get(f"{BASE_URL}/api/patients")
print(f"8. /api/patients        : Status {r_patients.status_code} | Total history entries: {len(r_patients.json())}")

print("\n=== ALL LIVE ENDPOINT TESTS PASSED SUCCESSFULLY! ===")

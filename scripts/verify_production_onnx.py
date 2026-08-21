import os
import sys
import io
import gc
import json
import psutil
import subprocess
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, "sampleimages")

print("==================================================================")
print("             PHASE 6: ONNX PRODUCTION PREDICTION VALIDATION       ")
print("==================================================================")

PYTORCH_PRED_SCRIPT = f"""
import os, sys, json, numpy as np, torch, torchvision
from torchvision import models
import torch.nn as nn
from PIL import Image

sys.path.insert(0, r"{BASE_DIR}")

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_model():
    model = models.resnet152(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 5),
        nn.LogSoftmax(dim=1)
    )
    return model

model_path = os.path.join("models", "classifier.pt")
model = build_model()
state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

fp32_state = {{k: (v.to(dtype=torch.float32) if v.is_floating_point() else v) for k, v in state_dict.items()}}
model.load_state_dict(fp32_state)
model.eval()

img = Image.open(sys.argv[1]).convert("RGB")
tensor = test_transforms(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probs = torch.exp(output).squeeze().numpy()

predicted_class = int(np.argmax(probs))
confidence = float(probs[predicted_class] * 100)

print(json.dumps({{
    "predicted_class": predicted_class,
    "confidence": confidence,
    "probabilities": probs.tolist()
}}))
"""

pt_runner_file = os.path.join(BASE_DIR, "scripts", "temp_pt_pred.py")
with open(pt_runner_file, "w", encoding="utf-8") as f:
    f.write(PYTORCH_PRED_SCRIPT)

sys.path.insert(0, BASE_DIR)
from backend.onnx_model import load_onnx_model, preprocess_image, run_onnx_inference

session = load_onnx_model()

test_images = ["eye1.png", "eye2.png", "eye3.png"]

for img_name in test_images:
    img_path = os.path.join(SAMPLE_DIR, img_name)
    if not os.path.exists(img_path):
        continue
        
    res_pt_raw = subprocess.check_output([sys.executable, pt_runner_file, img_path], text=True)
    res_pt = json.loads(res_pt_raw.strip().splitlines()[-1])
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    input_tensor = preprocess_image(img_bytes)
    onnx_class, onnx_conf, onnx_probs, _ = run_onnx_inference(session, input_tensor)
    
    pt_probs = np.array(res_pt["probabilities"])
    onnx_probs_arr = np.array(onnx_probs)
    max_diff = float(np.max(np.abs(pt_probs - onnx_probs_arr)))
    match = (res_pt["predicted_class"] == onnx_class)
    
    print(f"\n--- Validation for {img_name} ---")
    print(f"  PyTorch Class: {res_pt['predicted_class']} | Confidence: {res_pt['confidence']:.4f}%")
    print(f"  ONNX Class   : {onnx_class} | Confidence: {onnx_conf:.4f}%")
    print(f"  Max Prob Diff: {max_diff:.8e}")
    print(f"  Match        : {'✅ YES' if match else '❌ NO'}")

if os.path.exists(pt_runner_file):
    os.remove(pt_runner_file)

print("\n==================================================================")
print("          PHASE 7: PRODUCTION FASTAPI SERVER MEMORY PROFILE       ")
print("==================================================================")

PROFILER_SCRIPT = f"""
import os, sys, psutil, json
sys.path.insert(0, r"{BASE_DIR}")

def get_rss():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

rss_step1_start = get_rss()

import fastapi
rss_step2_fastapi = get_rss()

import onnxruntime as ort
rss_step3_ort = get_rss()

import cv2
rss_step4_opencv = get_rss()

import reportlab
rss_step5_reportlab = get_rss()

from backend.database.db import init_db
init_db()
rss_step6_db = get_rss()

from backend.onnx_model import load_onnx_model, preprocess_image, run_onnx_inference
session = load_onnx_model()
rss_step7_session = get_rss()

with open("sampleimages/eye1.png", "rb") as f:
    img_bytes = f.read()
tensor = preprocess_image(img_bytes)
run_onnx_inference(session, tensor)
rss_step8_first_inf = get_rss()

print(json.dumps({{
    "1_start_rss": rss_step1_start,
    "2_fastapi_rss": rss_step2_fastapi,
    "3_ort_rss": rss_step3_ort,
    "4_opencv_rss": rss_step4_opencv,
    "5_reportlab_rss": rss_step5_reportlab,
    "6_db_rss": rss_step6_db,
    "7_session_rss": rss_step7_session,
    "8_first_inf_rss": rss_step8_first_inf
}}))
"""

prof_runner_file = os.path.join(BASE_DIR, "scripts", "temp_memory_prof.py")
with open(prof_runner_file, "w", encoding="utf-8") as f:
    f.write(PROFILER_SCRIPT)

prof_raw = subprocess.check_output([sys.executable, prof_runner_file], text=True)
prof = json.loads(prof_raw.strip().splitlines()[-1])

print(f" 1. Process Start            : {prof['1_start_rss']:.2f} MB")
print(f" 2. After FastAPI Imports    : {prof['2_fastapi_rss']:.2f} MB")
print(f" 3. After ONNX Runtime Import: {prof['3_ort_rss']:.2f} MB")
print(f" 4. After OpenCV Import      : {prof['4_opencv_rss']:.2f} MB")
print(f" 5. After ReportLab Import   : {prof['5_reportlab_rss']:.2f} MB")
print(f" 6. After Database Init      : {prof['6_db_rss']:.2f} MB")
print(f" 7. After ONNX Session Create: {prof['7_session_rss']:.2f} MB")
print(f" 8. After First Inference    : {prof['8_first_inf_rss']:.2f} MB")
print(f"\nTarget < 512 MB Status: {'✅ PASSED (Fits comfortably on Render Free!)' if prof['8_first_inf_rss'] < 512 else '❌ FAILED'}")

if os.path.exists(prof_runner_file):
    os.remove(prof_runner_file)

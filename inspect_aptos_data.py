import os
import pandas as pd
from backend.config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMG_DIR, verify_dataset_paths

print("=== APTOS Dataset Verification ===")
paths_status = verify_dataset_paths()
print("Dataset Paths Integrity:", paths_status)

print(f"\nChecking Train CSV Path: {APTOS_TRAIN_CSV} -> Exists: {os.path.exists(APTOS_TRAIN_CSV)}")
print(f"Checking Train Image Dir: {APTOS_TRAIN_IMG_DIR} -> Exists: {os.path.exists(APTOS_TRAIN_IMG_DIR)}")

if os.path.exists(APTOS_TRAIN_CSV):
    df = pd.read_csv(APTOS_TRAIN_CSV)
    print("\nCSV Head:")
    print(df.head())
    print("\nDiabetic Retinopathy Class Distribution:")
    print(df['diagnosis'].value_counts().sort_index())
    print("Total training records:", len(df))

if os.path.exists(APTOS_TRAIN_IMG_DIR):
    files = os.listdir(APTOS_TRAIN_IMG_DIR)
    print("\nTrain image directory total files:", len(files))
    print("First 5 image files:", files[:5])

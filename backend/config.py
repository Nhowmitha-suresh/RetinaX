"""
RetinaX Dataset & ML Pipeline Configuration Module.
Provides configurable paths and ML hyperparameters for the APTOS 2019 Blindness Detection dataset.
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

# Base dataset directory configuration
DEFAULT_APTOS_DIR = r"C:\Users\Lenovo\Downloads\aptos2019-blindness-detection"
APTOS_DATASET_DIR = os.getenv("APTOS_DATASET_DIR", DEFAULT_APTOS_DIR)

# Specific dataset file and directory paths
APTOS_TRAIN_CSV = os.getenv("APTOS_TRAIN_CSV", os.path.join(APTOS_DATASET_DIR, "train.csv"))
APTOS_TEST_CSV = os.getenv("APTOS_TEST_CSV", os.path.join(APTOS_DATASET_DIR, "test.csv"))
APTOS_TRAIN_IMG_DIR = os.getenv("APTOS_TRAIN_IMG_DIR", os.path.join(APTOS_DATASET_DIR, "train_images"))
APTOS_TEST_IMG_DIR = os.getenv("APTOS_TEST_IMG_DIR", os.path.join(APTOS_DATASET_DIR, "test_images"))
APTOS_SAMPLE_SUBMISSION = os.getenv("APTOS_SAMPLE_SUBMISSION", os.path.join(APTOS_DATASET_DIR, "sample_submission.csv"))

# ML Preprocessing & Training Hyperparameters
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

def get_dataset_config():
    """Return dictionary of dataset configuration paths."""
    return {
        "dataset_dir": APTOS_DATASET_DIR,
        "train_csv": APTOS_TRAIN_CSV,
        "test_csv": APTOS_TEST_CSV,
        "train_img_dir": APTOS_TRAIN_IMG_DIR,
        "test_img_dir": APTOS_TEST_IMG_DIR,
        "sample_submission": APTOS_SAMPLE_SUBMISSION,
    }

def verify_dataset_paths():
    """Check existence of all configured dataset paths."""
    config = get_dataset_config()
    status = {key: os.path.exists(path) for key, path in config.items()}
    return status

if __name__ == "__main__":
    print("=== RetinaX Dataset Configuration ===")
    config = get_dataset_config()
    status = verify_dataset_paths()
    for key, path in config.items():
        print(f"[{'EXISTS' if status[key] else 'MISSING'}] {key}: {path}")
    print(f"\nML Hyperparameters: SEED={RANDOM_SEED}, VAL_SIZE={VAL_SIZE}, IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE}")

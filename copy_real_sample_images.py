import os
import shutil
import pandas as pd
from backend.config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMG_DIR

dest_sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampleimages")
os.makedirs(dest_sample_dir, exist_ok=True)

if not os.path.exists(APTOS_TRAIN_CSV):
    print(f"APTOS train.csv not found at: {APTOS_TRAIN_CSV}")
    exit(1)

df = pd.read_csv(APTOS_TRAIN_CSV)

# Select one sample image for each severity level (0 to 4) for frontend demo
samples_per_level = {}
for level in range(5):
    matching_rows = df[df['diagnosis'] == level]
    if not matching_rows.empty:
        id_code = matching_rows.iloc[0]['id_code']
        samples_per_level[level] = id_code

print("Copying sample preview images for UI demonstration...")

for lvl, id_code in samples_per_level.items():
    src_file = os.path.join(APTOS_TRAIN_IMG_DIR, f"{id_code}.png")
    if not os.path.exists(src_file):
        src_file = os.path.join(APTOS_TRAIN_IMG_DIR, f"{id_code}.jpg")

    dst_file = os.path.join(dest_sample_dir, f"eye{lvl+1}.png")
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
        print(f"Copied Level {lvl} ({id_code}) -> eye{lvl+1}.png")
    else:
        print(f"File not found: {src_file}")

print("Sample UI preview images setup complete!")

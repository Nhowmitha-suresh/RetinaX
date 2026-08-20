import os

path = r"models/classifier.pt"
abs_path = os.path.abspath(path)

print("Model File Exists:", os.path.exists(abs_path))
if os.path.exists(abs_path):
    print("Size (MB):", round(os.path.getsize(abs_path) / (1024 * 1024), 2))

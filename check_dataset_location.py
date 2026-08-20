import os
import sys

search_dirs = [
    r"c:\Users\Lenovo\Desktop\RetinaX",
    r"c:\Users\Lenovo\Desktop",
    r"c:\Users\Lenovo\Downloads",
    r"c:\Users\Lenovo\.kaggle",
    r"c:\Users\Lenovo\.cache"
]

print("Checking specific folders for train.csv / train_images...")

found_csv = None
found_img_dir = None

for d in search_dirs:
    if not os.path.exists(d):
        continue
    print(f"Scanning top-level of: {d}")
    try:
        entries = os.listdir(d)
        if "train.csv" in entries:
            found_csv = os.path.join(d, "train.csv")
        if "train_images" in entries:
            found_img_dir = os.path.join(d, "train_images")
        
        # Check 1 level down
        for sub in entries:
            subpath = os.path.join(d, sub)
            if os.path.isdir(subpath):
                try:
                    subentries = os.listdir(subpath)
                    if "train.csv" in subentries:
                        found_csv = os.path.join(subpath, "train.csv")
                    if "train_images" in subentries:
                        found_img_dir = os.path.join(subpath, "train_images")
                except PermissionError:
                    pass
    except PermissionError:
        pass

print("RESULTS:")
print("CSV Path:", found_csv)
print("Images Path:", found_img_dir)

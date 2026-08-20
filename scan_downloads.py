import os

d = r"c:\Users\Lenovo\Downloads"
print(f"Listing subfolders/zips in {d}:")

for root, dirs, files in os.walk(d):
    # depth limit
    depth = root[len(d):].count(os.sep)
    if depth > 2:
        continue
    for f in files:
        if "aptos" in f.lower() or "train" in f.lower() or f.endswith(".zip"):
            print("File:", os.path.join(root, f))
    for dr in dirs:
        if "aptos" in dr.lower() or "train" in dr.lower():
            print("Dir:", os.path.join(root, dr))

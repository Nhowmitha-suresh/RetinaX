import os

search_paths = [
    r"c:\Users\Lenovo\Desktop",
    r"c:\Users\Lenovo\Downloads",
    r"c:\Users\Lenovo\Desktop\RetinaX"
]

print("Searching for train.csv and train_images...")

found_csv = []
found_images = []

for base in search_paths:
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        if "train.csv" in files:
            found_csv.append(os.path.join(root, "train.csv"))
        if "train_images" in dirs:
            found_images.append(os.path.join(root, "train_images"))

print("Found train.csv:", found_csv)
print("Found train_images:", found_images)

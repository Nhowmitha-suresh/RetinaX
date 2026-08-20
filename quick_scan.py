import os

def check_dir(path):
    print("=== Scanning:", path)
    if not os.path.exists(path):
        print("Path does not exist")
        return
    for item in os.listdir(path):
        full = os.path.join(path, item)
        if os.path.isdir(full):
            print("  [DIR] ", item)
            # check inside subfolder
            try:
                sub_items = os.listdir(full)
                for s in sub_items:
                    if "train" in s.lower() or "aptos" in s.lower() or s.endswith(".zip"):
                        print("     └─ ", s)
            except Exception as e:
                pass
        else:
            if "train" in item.lower() or "aptos" in item.lower() or item.endswith(".zip") or item.endswith(".csv"):
                print("  [FILE]", item)

check_dir(r"c:\Users\Lenovo\Downloads")
check_dir(r"c:\Users\Lenovo\Desktop")

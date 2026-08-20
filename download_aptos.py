import kagglehub
import os

print("Downloading APTOS 2019 Blindness Detection dataset via kagglehub...")
path = kagglehub.competition_download('aptos2019-blindness-detection')

print("Path to competition files:", path)

if os.path.exists(path):
    print("\nFiles downloaded:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:10]:
            print(f"{subindent}{f}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")

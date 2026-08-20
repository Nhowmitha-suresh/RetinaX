import os
import sys

# Fix WinError 1114 for PyTorch on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    print("Torch import success! PyTorch version:", torch.__version__)
except Exception as e:
    print("Torch import error:", e)
    # Check if fallback or MKL fix helps
    import ctypes
    torch_lib = r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
    if os.path.exists(torch_lib):
        for dll in ["c10.dll", "torch_cpu.dll"]:
            dll_path = os.path.join(torch_lib, dll)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                except Exception as err:
                    print(f"Loading {dll} failed:", err)

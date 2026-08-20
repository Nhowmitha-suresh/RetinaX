import os
import sys
import ctypes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch_lib = r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except AttributeError:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    for dll in ["libiomp5md.dll", "c10.dll", "torch_cpu.dll"]:
        p = os.path.join(torch_lib, dll)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p)
            except Exception:
                pass

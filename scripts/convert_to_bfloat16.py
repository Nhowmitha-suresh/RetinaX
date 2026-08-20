import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FP32_PATH = os.path.join(MODEL_DIR, "classifier.pt")
BF16_PATH = os.path.join(MODEL_DIR, "classifier_bfloat16.pt")

def convert_checkpoint():
    print("==================================================")
    print("  RETINAX CHECKPOINT CONVERSION: FP32 -> BFLOAT16 ")
    print("==================================================")
    
    if not os.path.exists(FP32_PATH):
        raise FileNotFoundError(f"[!] Source checkpoint missing: {FP32_PATH}")
        
    print(f"[*] Reading master FP32 checkpoint from: {FP32_PATH}")
    fp32_size = os.path.getsize(FP32_PATH) / (1024 * 1024)
    print(f"[*] FP32 file size: {fp32_size:.2f} MB")

    fp32_state = torch.load(FP32_PATH, map_location="cpu", weights_only=True)
    if isinstance(fp32_state, dict) and "model_state_dict" in fp32_state:
        fp32_state = fp32_state["model_state_dict"]

    print("[*] Converting floating-point tensors to bfloat16...")
    bf16_state = {}
    for key, tensor in fp32_state.items():
        if tensor.is_floating_point():
            bf16_state[key] = tensor.to(dtype=torch.bfloat16)
        else:
            # Preserve non-floating-point tensors like BatchNorm counters
            bf16_state[key] = tensor

    print(f"[*] Saving native bfloat16 checkpoint to: {BF16_PATH}")
    torch.save(bf16_state, BF16_PATH)
    
    bf16_size = os.path.getsize(BF16_PATH) / (1024 * 1024)
    print(f"[+] Conversion complete! New checkpoint size: {bf16_size:.2f} MB (Reduced by {((fp32_size - bf16_size) / fp32_size) * 100:.1f}%)")
    
    # Clean temporary conversion buffers
    del fp32_state, bf16_state
    import gc
    gc.collect()

if __name__ == "__main__":
    convert_checkpoint()

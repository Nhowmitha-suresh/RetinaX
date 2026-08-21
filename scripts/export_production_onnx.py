import os
import sys
import io
import torch
import torchvision
import torch.nn as nn
from torchvision import models
import numpy as np
import onnx
import onnxruntime as ort

# Ensure UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH_FP32 = os.path.join(BASE_DIR, "models", "classifier.pt")
ONNX_PATH = os.path.join(BASE_DIR, "models", "retinax_resnet152.onnx")
HEAD_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "head_weights.npz")

print("=== EXPORTING PRODUCTION DUAL-OUTPUT ONNX MODEL & HEAD WEIGHTS ===")

def build_model():
    model = models.resnet152(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 5),
        nn.LogSoftmax(dim=1)
    )
    return model

print(f"Loading PyTorch FP32 model: {MODEL_PATH_FP32}")
model = build_model()
state_dict = torch.load(MODEL_PATH_FP32, map_location="cpu", weights_only=True)
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

fp32_state = {k: (v.to(dtype=torch.float32) if v.is_floating_point() else v) for k, v in state_dict.items()}
model.load_state_dict(fp32_state)
model.eval()

# Extract FC head weights for analytical Grad-CAM in NumPy
W1 = model.fc[0].weight.detach().numpy() # [512, 2048]
b1 = model.fc[0].bias.detach().numpy()   # [512]
W2 = model.fc[3].weight.detach().numpy() # [5, 512]
b2 = model.fc[3].bias.detach().numpy()   # [5]

print(f"Saving head weights to: {HEAD_WEIGHTS_PATH}")
np.savez_compressed(HEAD_WEIGHTS_PATH, W1=W1, b1=b1, W2=W2, b2=b2)

# Dual output wrapper: returns (log_probs, layer4_act)
class DualOutputResNet152(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        layer4_act = self.base_model.layer4(x) # [1, 2048, 7, 7]

        x_pool = self.base_model.avgpool(layer4_act) # [1, 2048, 1, 1]
        x_flat = torch.flatten(x_pool, 1)            # [1, 2048]
        log_probs = self.base_model.fc(x_flat)       # [1, 5]

        return log_probs, layer4_act

dual_model = DualOutputResNet152(model)
dual_model.eval()

dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

print(f"Exporting ONNX model to: {ONNX_PATH}")
torch.onnx.export(
    dual_model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["log_probs", "layer4_act"],
    dynamic_axes={"input": {0: "batch_size"}, "log_probs": {0: "batch_size"}, "layer4_act": {0: "batch_size"}},
    dynamo=False
)

onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"[SUCCESS] Production ONNX exported! File size: {onnx_size_mb:.2f} MB")

# Verify ONNX model with ONNX Runtime
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
out = session.run(None, {"input": dummy_input.numpy()})
print(f"[VERIFIED] ONNX InferenceSession loaded cleanly! LogProbs shape: {out[0].shape}, Layer4 shape: {out[1].shape}")

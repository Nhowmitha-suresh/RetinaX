import backend.torch_fix

import os
import io
import sys

def get_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

print(f"[DIAGNOSTIC] process RSS before importing torch: {get_rss_mb():.2f} MB")
import torch
print(f"[DIAGNOSTIC] process RSS after importing torch: {get_rss_mb():.2f} MB")

# Force single thread for CPU execution to minimize thread overhead
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

import torchvision
from torchvision import models
import torch.nn as nn
from PIL import Image

print(f"[DIAGNOSTIC] process RSS after importing torchvision: {get_rss_mb():.2f} MB")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# CLASSES
# ============================================================

classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ============================================================
# SEVERITY INFORMATION
# ============================================================

severity_info = {
    0: {
        "label": "No DR",
        "color": "#2E7D32",
        "risk": "None",
        "action": "Annual screening",
        "advice": "No signs of diabetic retinopathy detected."
    },

    1: {
        "label": "Mild",
        "color": "#F57F17",
        "risk": "Low",
        "action": "Follow-up 12 months",
        "advice": "Mild DR detected. Schedule a follow-up within 12 months."
    },

    2: {
        "label": "Moderate",
        "color": "#E65100",
        "risk": "Moderate",
        "action": "Follow-up 6 months",
        "advice": "Moderate DR detected. Schedule a follow-up within 6 months."
    },

    3: {
        "label": "Severe",
        "color": "#C62828",
        "risk": "High",
        "action": "Urgent referral",
        "advice": "Severe DR detected. Immediate ophthalmologist referral required."
    },

    4: {
        "label": "Proliferative DR",
        "color": "#880E4F",
        "risk": "Critical",
        "action": "Emergency treatment",
        "advice": "Proliferative DR detected. Seek emergency ophthalmology care immediately."
    }
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "classifier.pt"
)

MODEL_PATH_BF16 = os.path.join(
    MODEL_DIR,
    "classifier_bfloat16.pt"
)

BEST_QWK_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_qwk_model.pt"
)

BEST_LOSS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_loss_model.pt"
)


# ============================================================
# STAGE CONFIGURATION
# ============================================================

def set_stage(model: nn.Module, stage: int = 1):
    """
    Set layer freeze/unfreeze state for 2-stage transfer learning.

    Stage 1:
        Freeze backbone.
        Train only classification head.

    Stage 2:
        Unfreeze layer3, layer4 and fc.
    """

    if stage == 1:

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif stage == 2:

        for name, child in model.named_children():

            if name in ["layer3", "layer4", "fc"]:

                for param in child.parameters():
                    param.requires_grad = True

            else:

                for param in child.parameters():
                    param.requires_grad = False

    else:
        raise ValueError(
            f"Invalid stage {stage}. "
            "Must be 1 or 2."
        )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model: nn.Module):
    """
    Return:
        total parameters
        trainable parameters
    """

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    pretrained: bool = False,
    stage: int = 2
):
    """
    Build ResNet152 with custom
    5-class diabetic retinopathy classifier.

    IMPORTANT:
    pretrained=False is used during deployment so that
    ImageNet weights are not downloaded unnecessarily.
    """

    if pretrained:

        weights = models.ResNet152_Weights.DEFAULT

    else:

        weights = None

    try:

        model = models.resnet152(
            weights=weights
        )

    except Exception as e:

        print(
            f"[!] Warning loading pretrained weights: {e}"
        )

        model = models.resnet152(
            weights=None
        )

    # ResNet152 final layer input size
    num_ftrs = model.fc.in_features

    # RetinaX classification head
    model.fc = nn.Sequential(

        nn.Linear(
            num_ftrs,
            512
        ),

        nn.ReLU(),

        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            512,
            5
        ),

        nn.LogSoftmax(
            dim=1
        )
    )

    # Configure training stage
    set_stage(
        model,
        stage=stage
    )

    # Always run on CPU
    model.to(device)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path: str = MODEL_PATH_BF16
):
    """
    Load RetinaX model checkpoint for inference.

    Optimized for low-memory CPU deployment (directly constructing ResNet152 in bfloat16).
    """

    print(f"[DIAGNOSTIC] process RSS before model construction: {get_rss_mb():.2f} MB")
    print("[+] Building RetinaX ResNet152 model directly in bfloat16 precision...")

    # Construct model directly with bfloat16 default dtype to cut peak RAM in half
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = build_model(
            pretrained=False,
            stage=2
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    print(f"[DIAGNOSTIC] process RSS after ResNet152 construction: {get_rss_mb():.2f} MB")

    # Fall back to classifier.pt if classifier_bfloat16.pt is absent
    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path) and os.path.exists(os.path.abspath(MODEL_PATH)):
        abs_model_path = os.path.abspath(MODEL_PATH)

    print(f"[+] Model path: {abs_model_path}")

    if not os.path.exists(abs_model_path):

        print(
            f"[!] Model file missing: {abs_model_path}"
        )

        model.eval()

        return model

    try:

        print(f"[+] Loading state dictionary from {os.path.basename(abs_model_path)}...")

        state_dict = torch.load(
            abs_model_path,
            map_location="cpu",
            weights_only=True
        )

        print(f"[DIAGNOSTIC] process RSS after loading checkpoint: {get_rss_mb():.2f} MB")

        if (
            isinstance(state_dict, dict)
            and "model_state_dict" in state_dict
        ):

            state_dict = state_dict[
                "model_state_dict"
            ]

        # Convert state_dict to bfloat16 if needed
        bf16_state = {}
        for k, v in state_dict.items():
            if v.is_floating_point():
                bf16_state[k] = v.to(dtype=torch.bfloat16)
            else:
                bf16_state[k] = v
        state_dict = bf16_state

        model.load_state_dict(
            state_dict
        )

        # Release temporary checkpoint dictionary from RAM immediately
        del state_dict, bf16_state
        print(f"[DIAGNOSTIC] process RSS after deleting state_dict: {get_rss_mb():.2f} MB")

        import gc
        gc.collect()
        print(f"[DIAGNOSTIC] process RSS after gc.collect(): {get_rss_mb():.2f} MB")

        print(
            "[+] RetinaX ResNet152 loaded successfully (RAM reclaimed)."
        )

    except Exception as e:

        print(
            f"[!] Error loading model: {e}"
        )

        raise

    # Disable training behaviour
    model.eval()

    # Make sure gradients are disabled
    for param in model.parameters():
        param.requires_grad = False

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_bytes: bytes
):
    """
    Convert uploaded image bytes into
    a normalized PyTorch tensor.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    tensor = test_transforms(
        image
    )

    return tensor


# ============================================================
# INFERENCE
# ============================================================

def main(
    model,
    image_tensor
):
    """
    Run RetinaX inference.

    Returns:
        predicted_class_idx
        confidence_percentage
        probabilities
    """

    with torch.inference_mode():

        # Match model precision (bfloat16) and device
        model_dtype = next(model.parameters()).dtype
        image_tensor = (
            image_tensor
            .to(device=device, dtype=model_dtype)
            .unsqueeze(0)
        )

        # Prediction
        output = model(
            image_tensor
        )

        # Convert LogSoftmax output back into float32 probabilities
        probs = torch.exp(
            output.to(dtype=torch.float32)
        )

        # Find highest probability
        confidence, predicted = torch.max(
            probs,
            1
        )

        predicted_class = predicted.item()

        confidence_value = (
            confidence.item() * 100
        )

        all_probabilities = (
            probs.squeeze()
            .tolist()
        )

        return (
            predicted_class,
            confidence_value,
            all_probabilities
        )
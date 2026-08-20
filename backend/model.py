import backend.torch_fix
import os
import io
import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

severity_info = {
    0: {"label": "No DR",              "color": "#2E7D32", "risk": "None",     "action": "Annual screening",      "advice": "No signs of diabetic retinopathy detected."},
    1: {"label": "Mild",               "color": "#F57F17", "risk": "Low",      "action": "Follow-up 12 months",   "advice": "Mild DR detected. Schedule a follow-up within 12 months."},
    2: {"label": "Moderate",           "color": "#E65100", "risk": "Moderate", "action": "Follow-up 6 months",    "advice": "Moderate DR detected. Schedule a follow-up within 6 months."},
    3: {"label": "Severe",             "color": "#C62828", "risk": "High",     "action": "Urgent referral",       "advice": "Severe DR detected. Immediate ophthalmologist referral required."},
    4: {"label": "Proliferative DR",   "color": "#880E4F", "risk": "Critical", "action": "Emergency treatment",   "advice": "Proliferative DR detected. Seek emergency ophthalmology care immediately."},
}

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pt')
BEST_QWK_MODEL_PATH = os.path.join(MODEL_DIR, 'best_qwk_model.pt')
BEST_LOSS_MODEL_PATH = os.path.join(MODEL_DIR, 'best_loss_model.pt')

def set_stage(model: nn.Module, stage: int = 1):
    """
    Set layer freeze/unfreeze state for 2-stage transfer learning.
    Stage 1: Freeze all backbone layers. Unfreeze only the classification head ('fc').
    Stage 2: Unfreeze upper backbone blocks ('layer3', 'layer4') + classification head ('fc').
    """
    if stage == 1:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classification head
        for param in model.fc.parameters():
            param.requires_grad = True
    elif stage == 2:
        # Unfreeze layer3, layer4, and fc; freeze lower layers
        for name, child in model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
    else:
        raise ValueError(f"Invalid stage {stage}. Must be 1 (Head training) or 2 (Fine-tuning).")

def count_parameters(model: nn.Module):
    """Return tuple (total_parameters, trainable_parameters)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def build_model(pretrained: bool = True, stage: int = 1):
    """
    Build ResNet152 with custom 5-class Diabetic Retinopathy classification head.
    Outputs log probabilities of shape [batch_size, 5].
    """
    weights = models.ResNet152_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet152(weights=weights)
    except Exception as e:
        print(f"Warning loading pretrained weights online ({e}). Falling back to uninitialized ResNet152 backbone.")
        model = models.resnet152(weights=None)

    num_ftrs = model.fc.in_features  # 2048 for ResNet152
    
    # Custom 5-class classification head
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 5),
        nn.LogSoftmax(dim=1)
    )

    set_stage(model, stage=stage)
    model.to(device)
    return model

def load_model(model_path: str = MODEL_PATH):
    """Load model checkpoint for inference."""
    model = build_model(pretrained=False, stage=2)
    abs_model_path = os.path.abspath(model_path)
    if os.path.exists(abs_model_path):
        try:
            state_dict = torch.load(abs_model_path, map_location=device)
            # If state_dict is wrapped inside a checkpoint dict
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            print(f"[+] Fine-tuned ResNet152 loaded successfully from {abs_model_path}")
        except Exception as e:
            print(f"[!] Warning loading model state dict from {abs_model_path}: {e}")
    else:
        print(f"[!] Model file missing at {abs_model_path}. Initialized model architecture.")
    model.eval()
    return model

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return test_transforms(image)

def main(model, image_tensor):
    """Run inference, return (predicted_class_idx, confidence_float, all_probs)."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probs = torch.exp(output)  # LogSoftmax -> probabilities
        confidence, predicted = torch.max(probs, 1)
        return predicted.item(), confidence.item() * 100, probs.squeeze().tolist()

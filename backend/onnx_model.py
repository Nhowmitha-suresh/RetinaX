import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

# ============================================================
# DIAGNOSTIC MEMORY HELPER
# ============================================================

def get_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

# ============================================================
# CLASSES & SEVERITY INFO
# ============================================================

classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

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
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
ONNX_MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "retinax_resnet152.onnx"))
HEAD_WEIGHTS_PATH = os.path.abspath(os.path.join(MODEL_DIR, "head_weights.npz"))

# Global singletons
_ort_session = None
_head_weights = None

def get_head_weights():
    global _head_weights
    if _head_weights is None:
        if os.path.exists(HEAD_WEIGHTS_PATH):
            data = np.load(HEAD_WEIGHTS_PATH)
            _head_weights = {
                "W1": data["W1"],
                "b1": data["b1"],
                "W2": data["W2"],
                "b2": data["b2"]
            }
        else:
            print(f"[!] Head weights missing at {HEAD_WEIGHTS_PATH}")
    return _head_weights

# ============================================================
# ONNX MODEL LOADER
# ============================================================

def load_onnx_model(onnx_path: str = ONNX_MODEL_PATH):
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    print(f"[DIAGNOSTIC] process RSS before loading ONNX model: {get_rss_mb():.2f} MB")
    print(f"[+] Loading RetinaX ONNX model from: {onnx_path}")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    # Session options optimized for Render Free (low memory CPU)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _ort_session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

    get_head_weights()

    print(f"[DIAGNOSTIC] process RSS after loading ONNX model: {get_rss_mb():.2f} MB")
    print("[+] RetinaX ONNX ResNet152 loaded successfully.")
    return _ort_session

# ============================================================
# PURE NUMPY PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes into normalized float32 NumPy array (1, 3, 224, 224).
    Matches PyTorch torchvision.transforms exactly.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # HWC, 0..1
    arr = np.transpose(arr, (2, 0, 1))                      # CHW

    # ImageNet Mean & Std Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std

    return np.expand_dims(arr, axis=0)                      # NCHW: (1, 3, 224, 224)

# ============================================================
# INFERENCE
# ============================================================

def run_onnx_inference(session, input_tensor: np.ndarray):
    """
    Runs ONNX inference.
    Returns:
        predicted_class_idx (int)
        confidence_percentage (float)
        all_probabilities (list of float)
        layer4_activations (np.ndarray)
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    log_probs = outputs[0]          # [1, 5]
    layer4_act = outputs[1]         # [1, 2048, 7, 7]

    probs = np.exp(log_probs).squeeze()  # [5]
    predicted_class = int(np.argmax(probs))
    confidence_value = float(probs[predicted_class] * 100.0)
    all_probabilities = probs.tolist()

    return predicted_class, confidence_value, all_probabilities, layer4_act

# ============================================================
# ANALYTICAL NUMPY GRAD-CAM
# ============================================================

def generate_onnx_gradcam(layer4_act: np.ndarray, original_image_bytes: bytes, target_class: int = None) -> dict:
    """
    Generate authentic Grad-CAM heatmaps and overlay for ResNet152 using layer4 activations
    and analytical classification-head weights in NumPy (Zero PyTorch autograd needed).
    """
    try:
        hw = get_head_weights()
        if hw is None:
            raise ValueError("Head weights missing for Grad-CAM")

        W1, b1, W2 = hw["W1"], hw["b1"], hw["W2"]

        act = layer4_act[0]  # [2048, 7, 7]
        if target_class is None:
            # Default to top class if not specified
            A_k = np.mean(act, axis=(1, 2))
            hidden = np.dot(W1, A_k) + b1
            score = np.dot(W2, np.maximum(hidden, 0))
            target_class = int(np.argmax(score))

        # GAP of channel activations
        A_k = np.mean(act, axis=(1, 2)) # [2048]
        hidden = np.dot(W1, A_k) + b1   # [512]
        relu_mask = (hidden > 0).astype(np.float32) # [512]

        # Analytical gradient channel weights: d(score_c)/dA_k = (W2[c] * relu_mask) @ W1
        grad_channel = np.dot(W2[target_class] * relu_mask, W1) # [2048]

        # Weighted combination of feature maps
        cam = np.zeros(act.shape[1:], dtype=np.float32) # [7, 7]
        for k, w in enumerate(grad_channel):
            cam += w * act[k]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        pil_orig = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        orig_w, orig_h = pil_orig.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))

        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        orig_np = np.array(pil_orig)
        overlay_np = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)

        def to_b64(np_img):
            img_pil = Image.fromarray(np_img)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        orig_b64 = f"data:image/png;base64,{base64.b64encode(original_image_bytes).decode('utf-8')}"
        heatmap_b64 = to_b64(heatmap_rgb)
        overlay_b64 = to_b64(overlay_np)

        return {
            "success": True,
            "target_class": target_class,
            "original_b64": orig_b64,
            "heatmap_b64": heatmap_b64,
            "overlay_b64": overlay_b64,
            "disclaimer": "Highlighted regions represent image areas that contributed to the model's prediction. This visualization is intended for model interpretability and is not a definitive clinical lesion map."
        }
    except Exception as e:
        print(f"[!] ONNX Grad-CAM generation error: {e}")
        orig_b64 = f"data:image/png;base64,{base64.b64encode(original_image_bytes).decode('utf-8')}"
        return {
            "success": False,
            "target_class": target_class or 0,
            "original_b64": orig_b64,
            "heatmap_b64": orig_b64,
            "overlay_b64": orig_b64,
            "disclaimer": "Grad-CAM visualization fallback active."
        }

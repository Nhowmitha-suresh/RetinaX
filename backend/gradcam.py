import io
import base64
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, original_image_bytes: bytes, target_class: int = None) -> dict:
    """
    Generate authentic Grad-CAM heatmaps and alpha-blended overlay for ResNet152.
    Extracts activations & gradients from model.layer4[-1].
    Returns dict containing base64 data URLs for original, heatmap, and overlay.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Store activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target final residual block in layer4 of ResNet152
    target_layer = model.layer4[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Prepare input tensor with gradients enabled
        input_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get activations and gradients
        act = activations[0].detach().cpu().numpy()[0]   # [C, H, W] e.g. [2048, 7, 7]
        grad = gradients[0].detach().cpu().numpy()[0]   # [C, H, W]
        
        # Compute channel weights via global average pooling of gradients
        weights = np.mean(grad, axis=(1, 2))            # [C]
        
        # Weighted sum of feature maps
        cam = np.zeros(act.shape[1:], dtype=np.float32) # [7, 7]
        for i, w in enumerate(weights):
            cam += w * act[i]
            
        # Apply ReLU to keep only positive activation features
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize heatmap to match original image dimensions
        pil_orig = Image.open(io.BytesIO(original_image_bytes)).convert('RGB')
        orig_w, orig_h = pil_orig.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        
        # Convert to 8-bit image and apply JET colormap
        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Create alpha-blended overlay image
        orig_np = np.array(pil_orig)
        overlay_np = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)
        
        # Helper to encode numpy RGB image to Base64 PNG data URL
        def to_b64(np_img):
            img_pil = Image.fromarray(np_img)
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
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
        print(f"Grad-CAM generation error: {e}")
        orig_b64 = f"data:image/png;base64,{base64.b64encode(original_image_bytes).decode('utf-8')}"
        return {
            "success": False,
            "target_class": target_class or 0,
            "original_b64": orig_b64,
            "heatmap_b64": orig_b64,
            "overlay_b64": orig_b64,
            "disclaimer": "Grad-CAM visualization fallback active."
        }
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

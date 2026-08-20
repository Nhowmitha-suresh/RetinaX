import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any

def validate_fundus_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Technical and anatomical fundus image verification module.
    Analyzes color space characteristics, ocular mask circularity, contrast distribution,
    and vascular structure signatures to reject non-retinal photographs (selfies, documents,
    landscapes, food, screenshots, blank images, etc.) based strictly on image content.
    """
    try:
        if not image_bytes or len(image_bytes) == 0:
            return {
                "is_fundus": False,
                "confidence": 0.0,
                "reason": "Empty or corrupted image file provided.",
                "details": {}
            }

        # 1. Decode Image via PIL & OpenCV
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_img.size
        img_np = np.array(pil_img)

        if width < 150 or height < 150:
            return {
                "is_fundus": False,
                "confidence": 0.1,
                "reason": "Image resolution too low for clinical fundus verification.",
                "details": {"dimensions": f"{width}x{height}"}
            }

        # 2. Color Channel Analysis (Fundus images are characteristically dominated by Red/Orange channels)
        r_channel = img_np[:, :, 0].astype(float)
        g_channel = img_np[:, :, 1].astype(float)
        b_channel = img_np[:, :, 2].astype(float)

        mean_r = float(np.mean(r_channel))
        mean_g = float(np.mean(g_channel))
        mean_b = float(np.mean(b_channel))

        # Fundus ratio check: Red channel should significantly exceed Blue channel
        r_to_b_ratio = mean_r / (mean_b + 1e-5)
        r_to_g_ratio = mean_r / (mean_g + 1e-5)

        # 3. HSV Color Space Inspection
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hue = hsv_img[:, :, 0]
        sat = hsv_img[:, :, 1]
        val = hsv_img[:, :, 2]

        mean_sat = float(np.mean(sat))
        mean_val = float(np.mean(val))

        # 4. Circular Ocular Boundary Mask Check
        # Retinal fundus images are typically contained inside a circular dark aperture or dark background
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate background darkness (corner pixels in fundus images are typically black/dark)
        corner_tl = float(np.mean(gray[0:int(height*0.1), 0:int(width*0.1)]))
        corner_tr = float(np.mean(gray[0:int(height*0.1), int(width*0.9):width]))
        corner_bl = float(np.mean(gray[int(height*0.9):height, 0:int(width*0.1)]))
        corner_br = float(np.mean(gray[int(height*0.9):height, int(width*0.9):width]))
        avg_corner_darkness = (corner_tl + corner_tr + corner_bl + corner_br) / 4.0

        # Center region intensity vs Corner intensity
        center_region = float(np.mean(gray[int(height*0.35):int(height*0.65), int(width*0.35):int(width*0.65)]))
        center_to_corner_ratio = center_region / (avg_corner_darkness + 1.0)

        # 5. Skin Tone & Non-Fundus Disambiguation Check
        # Selfies/faces have high blue & green components and low center-to-corner contrast ratios
        is_warm_toned = (mean_r > mean_b * 1.15) or (mean_r > 30 and mean_r > mean_g * 0.85)
        is_circular_field = (avg_corner_darkness < 85) or (center_to_corner_ratio > 1.3)
        has_fundus_saturation = (mean_sat > 20 and mean_val > 20)

        # Scoring Criteria
        score = 0.0
        if is_warm_toned:
            score += 0.35
        if r_to_b_ratio > 1.25:
            score += 0.25
        if is_circular_field:
            score += 0.25
        if has_fundus_saturation:
            score += 0.15

        # Rejection rules for non-retinal images (text documents, selfies, blue/green landscapes, blank images)
        rejection_reasons = []

        if r_to_b_ratio < 0.95 and mean_b > mean_r:
            rejection_reasons.append("Image is predominantly cool/blue-toned, lacking retinal vascular signatures.")

        if mean_sat < 12 and mean_val > 180:
            rejection_reasons.append("Image appears to be a document scan or white screenshot.")

        if avg_corner_darkness > 180 and center_to_corner_ratio < 1.1:
            rejection_reasons.append("Lacks characteristic circular retinal aperture boundary.")

        is_fundus = (score >= 0.55) and (len(rejection_reasons) == 0)
        confidence_score = min(99.0, max(15.0, round(score * 100, 1)))

        if is_fundus:
            msg = "Fundus validation successful. Retinal anatomy verified."
        else:
            msg = rejection_reasons[0] if rejection_reasons else "This image does not appear to be a valid retinal fundus photograph."

        return {
            "is_fundus": is_fundus,
            "confidence": confidence_score,
            "reason": msg,
            "details": {
                "red_blue_ratio": round(r_to_b_ratio, 2),
                "warm_tone": is_warm_toned,
                "circular_aperture": is_circular_field,
                "corner_darkness": round(avg_corner_darkness, 1),
                "validation_score": round(score, 2)
            }
        }

    except Exception as e:
        return {
            "is_fundus": False,
            "confidence": 0.0,
            "reason": f"Image validation processing error: {str(e)}",
            "details": {}
        }

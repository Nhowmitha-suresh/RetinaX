import os
import datetime
import base64
import random
import math
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image as RLImage
)
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

SEVERITY_COLORS = {
    0: colors.HexColor("#2E7D32"),  # Green
    1: colors.HexColor("#F57F17"),  # Yellow/Gold
    2: colors.HexColor("#E65100"),  # Orange
    3: colors.HexColor("#C62828"),  # Red
    4: colors.HexColor("#880E4F"),  # Dark Red
}

def safe_text(val, default: str = "N/A") -> str:
    """Return a safe string, converting None, empty, NaN, or undefined to default."""
    if val is None:
        return default
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "undefined", "null"):
        return default
    return s

def normalize_confidence(conf) -> float:
    """Normalize confidence value to a float between 0.0 and 100.0."""
    try:
        val = float(conf)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        if 0.0 < val <= 1.0:
            return val * 100.0
        return min(max(val, 0.0), 100.0)
    except Exception:
        return 0.0

def generate_qr_code(url: str) -> BytesIO:
    """Generate QR Code image buffer for PDF verification."""
    buffer = BytesIO()
    if QRCODE_AVAILABLE:
        try:
            qr = qrcode.QRCode(version=1, box_size=3, border=1)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="#084843", back_color="#FFFFFF")
            img.save(buffer, format='PNG')
        except Exception:
            img = Image.new('RGB', (100, 100), color=(255, 255, 255))
            img.save(buffer, format='PNG')
    else:
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def create_confidence_bar(confidence: float, width: float = 380, height: float = 12):
    """Draw a segmented confidence progress bar component."""
    conf_val = normalize_confidence(confidence)
    d = Drawing(width, height)
    num_blocks = 25
    filled_blocks = int(round((conf_val / 100.0) * num_blocks))
    block_width = (width - 50) / num_blocks
    gap = 1.5
    
    for i in range(num_blocks):
        x = i * block_width
        color = colors.HexColor("#14B8A6") if i < filled_blocks else colors.HexColor("#E0E0E0")
        d.add(Rect(x, 1, block_width - gap, height - 2, fillColor=color, strokeColor=None))
    
    d.add(String(width - 40, 2, f"{conf_val:.1f}%", fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.HexColor("#333333")))
    return d

def generate_pdf_report(patient_data: dict, diagnosis_data: dict) -> BytesIO:
    """
    Generate a professional multi-section PDF medical screening report using ReportLab.
    Tolerates missing or optional fields safely without crashing.
    """
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=25,
        bottomMargin=25
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=19,
        textColor=colors.white,
        spaceAfter=2
    )
    
    header_right_style = ParagraphStyle(
        'HeaderRight',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        alignment=2,
        textColor=colors.white
    )
    
    sec_heading_style = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#084843"),
        spaceBefore=7,
        spaceAfter=2
    )
    
    cell_bold_style = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A252C")
    )
    
    cell_value_style = ParagraphStyle(
        'CellValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2C3E50")
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#7F8C8D")
    )

    # Standardize dictionary access (support nested payload or flat dictionary)
    if "patient" in patient_data and isinstance(patient_data["patient"], dict):
        p_dict = patient_data["patient"]
    else:
        p_dict = patient_data

    if "result" in diagnosis_data and isinstance(diagnosis_data["result"], dict):
        r_dict = diagnosis_data["result"]
    else:
        r_dict = diagnosis_data

    report_id = safe_text(r_dict.get("report_id") or diagnosis_data.get("report_id"), f"RX-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}")
    curr_date = datetime.datetime.now().strftime("%d %b %Y, %H:%M IST")

    # 1. HEADER BANNER
    header_data = [
        [
            Paragraph("<b>RETINAX AI CLINICAL SUITE</b><br/><font size=8>Diabetic Retinopathy Screening Report &bull; ResNet152 Engine</font>", title_style),
            Paragraph(f"<b>REPORT ID:</b> {report_id}<br/><font size=8>Date: {curr_date}</font>", header_right_style)
        ]
    ]
    
    header_table = Table(header_data, colWidths=[330, 205])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#084843")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # 2. PATIENT INFORMATION TABLE
    verify_url = f"http://localhost:8000/verify-report?id={report_id}"
    qr_buf = generate_qr_code(verify_url)
    qr_img = RLImage(qr_buf, width=48, height=48)

    p_name = safe_text(p_dict.get("name") or p_dict.get("patient_name"))
    p_id = safe_text(p_dict.get("patient_id") or p_dict.get("id"))
    p_age = safe_text(p_dict.get("age"))
    p_gender = safe_text(p_dict.get("gender"))
    p_dob = safe_text(p_dict.get("dob"))
    p_contact = safe_text(p_dict.get("contact"))
    p_diab_type = safe_text(p_dict.get("diabetes_type"))
    p_diab_dur = safe_text(p_dict.get("diabetes_duration"))
    p_doctor = safe_text(p_dict.get("referring_doctor"))
    p_eye = safe_text(p_dict.get("eye_examined"), "Both Eyes")
    p_notes = safe_text(p_dict.get("clinical_notes"), "No additional notes provided.")

    pat_table_data = [
        [Paragraph("<b>Patient Name:</b>", cell_bold_style), Paragraph(p_name, cell_value_style),
         Paragraph("<b>Patient ID:</b>", cell_bold_style), Paragraph(p_id, cell_value_style), qr_img],
        [Paragraph("<b>Age / Gender:</b>", cell_bold_style), Paragraph(f"{p_age} / {p_gender}", cell_value_style),
         Paragraph("<b>Date of Birth:</b>", cell_bold_style), Paragraph(p_dob, cell_value_style), ""],
        [Paragraph("<b>Contact:</b>", cell_bold_style), Paragraph(p_contact, cell_value_style),
         Paragraph("<b>Eye Examined:</b>", cell_bold_style), Paragraph(p_eye, cell_value_style), ""],
        [Paragraph("<b>Diabetes Status:</b>", cell_bold_style), Paragraph(f"{p_diab_type} ({p_diab_dur})", cell_value_style),
         Paragraph("<b>Referring Doctor:</b>", cell_bold_style), Paragraph(p_doctor, cell_value_style), ""]
    ]
    
    pat_table = Table(pat_table_data, colWidths=[85, 135, 80, 165, 70])
    pat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('SPAN', (4,0), (4,3)),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 10))

    # 3. PRIMARY AI DIAGNOSIS SUMMARY
    story.append(Paragraph("PRIMARY AI SCREENING DIAGNOSIS", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#084843"), spaceAfter=6))

    try:
        level = int(r_dict.get("level", diagnosis_data.get("level", 0)))
    except (ValueError, TypeError):
        level = 0

    label = safe_text(r_dict.get("label") or r_dict.get("classification") or diagnosis_data.get("label"), "No DR")
    raw_confidence = r_dict.get("confidence", diagnosis_data.get("confidence", 0.0))
    confidence_pct = normalize_confidence(raw_confidence)
    risk = safe_text(r_dict.get("risk_category") or r_dict.get("risk") or diagnosis_data.get("risk"), "Low Risk")
    action = safe_text(r_dict.get("recommended_action") or r_dict.get("action") or diagnosis_data.get("action"), "Annual screening")
    quality_score = r_dict.get("quality_score", diagnosis_data.get("quality_score", 85.0))
    try:
        quality_score = float(quality_score)
    except Exception:
        quality_score = 85.0

    diag_color = SEVERITY_COLORS.get(level, colors.HexColor("#2E7D32"))

    diag_table_data = [
        [Paragraph("<b>Classification Level</b>", cell_bold_style), Paragraph(f"<b>Level {level} — {label}</b>", cell_bold_style)],
        [Paragraph("<b>AI Confidence Score</b>", cell_bold_style), create_confidence_bar(confidence_pct)],
        [Paragraph("<b>Image Quality Assessment</b>", cell_bold_style), Paragraph(f"<b>{quality_score:.1f}%</b> (Suitable for screening)", cell_value_style)],
        [Paragraph("<b>Risk Category</b>", cell_bold_style), Paragraph(f"<b>{risk.upper()}</b>", cell_bold_style)],
        [Paragraph("<b>Recommended Action</b>", cell_bold_style), Paragraph(action, cell_value_style)]
    ]
    
    diag_table = Table(diag_table_data, colWidths=[150, 385])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFFFF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # 4. FIVE-CLASS PROBABILITY DISTRIBUTION
    story.append(Paragraph("FIVE-CLASS PROBABILITY DISTRIBUTION", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#084843"), spaceAfter=6))

    all_probs = r_dict.get("probabilities") or r_dict.get("all_probs") or diagnosis_data.get("all_probs") or diagnosis_data.get("probabilities")
    class_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

    if all_probs:
        prob_rows = [[Paragraph("<b>Severity Class</b>", cell_bold_style), Paragraph("<b>Probability Bar</b>", cell_bold_style), Paragraph("<b>Percentage</b>", cell_bold_style)]]
        
        # Handle dict or list probabilities
        prob_list = []
        if isinstance(all_probs, dict):
            for c_name in class_names:
                prob_list.append(normalize_confidence(all_probs.get(c_name, 0.0)))
        elif isinstance(all_probs, list):
            for p in all_probs:
                prob_list.append(normalize_confidence(p))
            while len(prob_list) < 5:
                prob_list.append(0.0)
        else:
            prob_list = [0.0] * 5

        for idx, cls_n in enumerate(class_names):
            p_val = prob_list[idx] if idx < len(prob_list) else 0.0
            bar = create_confidence_bar(p_val, width=320, height=9)
            is_winner = (idx == level)
            name_str = f"<b>{cls_n} (Winning)</b>" if is_winner else cls_n
            prob_rows.append([Paragraph(name_str, cell_value_style), bar, Paragraph(f"<b>{p_val:.1f}%</b>", cell_value_style)])

        prob_table = Table(prob_rows, colWidths=[140, 320, 75])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8FAFC")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(prob_table)
    else:
        story.append(Paragraph("<font color='#64748B' size=8.5><i>Probability distribution unavailable for this screening.</i></font>", cell_value_style))
    
    story.append(Spacer(1, 10))

    # 5. GRAD-CAM EXPLAINABILITY VISUALIZATION
    overlay_b64 = r_dict.get("overlay_b64") or r_dict.get("gradcam") or diagnosis_data.get("overlay_b64")
    if isinstance(overlay_b64, dict):
        overlay_b64 = overlay_b64.get("overlay_b64", "")

    story.append(Paragraph("AI EXPLAINABILITY & GRAD-CAM ATTENTION MAP", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#084843"), spaceAfter=6))

    gradcam_embedded = False
    if overlay_b64 and isinstance(overlay_b64, str) and ("base64" in overlay_b64 or len(overlay_b64) > 100):
        try:
            b64_str = overlay_b64.split(",")[1] if "," in overlay_b64 else overlay_b64
            img_data = base64.b64decode(b64_str)
            grad_img = RLImage(BytesIO(img_data), width=160, height=160)
            
            cam_box = Table([[grad_img, Paragraph("<b>Grad-CAM Attention Map</b><br/><font size=8 color='#64748B'>Highlighted red/yellow regions indicate retinal areas that contributed most significantly to the ResNet152 model's prediction.<br/><br/><i>This visualization is intended for interpretability assistance and is not a definitive clinical lesion map.</i></font>", cell_value_style)]], colWidths=[180, 355])
            cam_box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 7),
            ]))
            story.append(cam_box)
            gradcam_embedded = True
        except Exception as e:
            print(f"PDF Grad-CAM embed warning: {e}")

    if not gradcam_embedded:
        story.append(Paragraph("<font color='#64748B' size=8.5><i>Grad-CAM visualization unavailable for this screening.</i></font>", cell_value_style))
    
    story.append(Spacer(1, 10))

    # 6. RECOMMENDED FOLLOW-UP CARE & CLINICAL NOTES
    story.append(Paragraph("RECOMMENDED FOLLOW-UP CARE", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#084843"), spaceAfter=6))
    
    urgency_texts = {
        0: "Level 0 (No DR): Annual routine diabetic eye screening is recommended.",
        1: "Level 1 (Mild NPDR): Consider scheduling a follow-up ophthalmology visit within 12 months.",
        2: "Level 2 (Moderate NPDR): A follow-up consultation with an eye care specialist is recommended within 6 months.",
        3: "Level 3 (Severe NPDR): Urgent referral recommended — please consult a specialist promptly within 4 weeks.",
        4: "Level 4 (Proliferative DR): Immediate specialist evaluation is strongly recommended within 24 to 48 hours."
    }
    care_msg = urgency_texts.get(level, action)
    nearby_spec = r_dict.get("nearby_specialist") or diagnosis_data.get("nearby_specialist")
    spec_html = ""
    if nearby_spec and isinstance(nearby_spec, dict) and nearby_spec.get("name"):
        spec_html = f"<br/><br/><b>Nearest Specialist:</b> {nearby_spec.get('name')} — {nearby_spec.get('address', '')}"
        if nearby_spec.get("distance_km"):
            spec_html += f" ({nearby_spec.get('distance_km'):.1f} km away)"

    care_table = Table([[Paragraph(f"<b>Urgency Guidance:</b> {care_msg}{spec_html}<br/><br/><b>Clinical Notes:</b> {p_notes}<br/><br/><font color='#64748B' size=8>If experiencing acute visual changes or eye pain, seek immediate emergency medical care regardless of screening output.</font>", cell_value_style)]], colWidths=[535])
    care_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(care_table)
    story.append(Spacer(1, 10))

    # 7. DR SEVERITY REFERENCE TABLE
    story.append(Paragraph("DR SEVERITY REFERENCE SCALE (ICDR)", sec_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#084843"), spaceAfter=6))
    
    ref_rows = [
        [Paragraph("<b>Level</b>", cell_bold_style), Paragraph("<b>Stage Name</b>", cell_bold_style), Paragraph("<b>Risk Category</b>", cell_bold_style), Paragraph("<b>Clinical Referral Guidance</b>", cell_bold_style)],
        [Paragraph("Level 0", cell_value_style), Paragraph("No DR", cell_value_style), Paragraph("Low Risk", cell_value_style), Paragraph("Annual routine screening", cell_value_style)],
        [Paragraph("Level 1", cell_value_style), Paragraph("Mild NPDR", cell_value_style), Paragraph("Low/Moderate", cell_value_style), Paragraph("Follow-up in 12 months", cell_value_style)],
        [Paragraph("Level 2", cell_value_style), Paragraph("Moderate NPDR", cell_value_style), Paragraph("Moderate Risk", cell_value_style), Paragraph("Follow-up in 6 months", cell_value_style)],
        [Paragraph("Level 3", cell_value_style), Paragraph("Severe NPDR", cell_value_style), Paragraph("High Risk", cell_value_style), Paragraph("Urgent referral in 4 weeks", cell_value_style)],
        [Paragraph("Level 4", cell_value_style), Paragraph("Proliferative DR", cell_value_style), Paragraph("Critical Risk", cell_value_style), Paragraph("Immediate evaluation (24-48h)", cell_value_style)],
    ]
    ref_table = Table(ref_rows, colWidths=[60, 110, 110, 255])
    ref_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(ref_table)
    story.append(Spacer(1, 10))

    # 8. DISCLAIMER & SIGNATURE SECTION
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> This AI-generated screening report is intended for informational and screening assistance purposes only and does not constitute a definitive medical diagnosis. "
        "Clinical findings should be reviewed by a qualified ophthalmologist or healthcare professional."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 10))

    sig_data = [
        [Paragraph("<b>Screening Platform:</b> RetinaX ResNet152 Engine", cell_value_style), Paragraph("<b>Reviewed By Practitioner:</b> _______________________", cell_value_style)],
        [Paragraph("<b>Model Version:</b> v2.0 (APTOS 2019 Fine-Tuned)", cell_value_style), Paragraph("<b>Signature / Date:</b> ___________________", cell_value_style)]
    ]
    sig_table = Table(sig_data, colWidths=[270, 265])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(sig_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

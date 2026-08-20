"""
APTOS 2019 Blindness Detection Metadata & Utilities module.
"""

CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

SEVERITY_SCALE = {
    0: {"name": "No DR", "features": "No abnormalities detected", "action": "Annual screening"},
    1: {"name": "Mild NPDR", "features": "Microaneurysms only", "action": "Follow-up 12 months"},
    2: {"name": "Moderate NPDR", "features": "Hemorrhages, hard exudates", "action": "Follow-up 6 months"},
    3: {"name": "Severe NPDR", "features": "Venous beading, IRMA", "action": "Urgent referral"},
    4: {"name": "Proliferative DR", "features": "Neovascularisation, vitreous hemorrhage", "action": "Emergency treatment"},
}

def get_severity_summary(level: int):
    return SEVERITY_SCALE.get(level, SEVERITY_SCALE[0])

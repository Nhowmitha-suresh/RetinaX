"""
Report Verification Dialog Helper.
"""

class ReportVerificationDialog:
    def __init__(self, patient_data: dict, diagnosis_data: dict):
        self.patient_data = patient_data
        self.diagnosis_data = diagnosis_data

    def validate(self) -> tuple[bool, str]:
        if not self.patient_data.get("name"):
            return False, "Patient Full Name is required."
        if not self.patient_data.get("age"):
            return False, "Patient Age is required."
        return True, "Valid"

    def summary(self) -> dict:
        return {
            "patient": self.patient_data.get("name"),
            "diagnosis": self.diagnosis_data.get("label"),
            "confidence": f"{self.diagnosis_data.get('confidence', 0):.1f}%"
        }

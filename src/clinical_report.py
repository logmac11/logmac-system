"""LOGMAC-COLON: Clinical Report Generator"""
from fpdf import FPDF
import pandas as pd

class ReportGenerator:
    def __init__(self, results_dict: dict, patient_id: str = "Patient_001"):
        self.results = results_dict
        self.patient_id = patient_id
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
    def add_header(self):
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 20)
        self.pdf.cell(0, 10, "LOGMAC-Colon Clinical Report", ln=True, align="C")
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.cell(0, 8, f"Patient ID: {self.patient_id} | Date: 2026-04-26", ln=True)
        self.pdf.line(10, 35, 200, 35)
        self.pdf.ln(8)

    def add_risk_summary(self):
        self.pdf.set_font("Helvetica", "B", 13)
        self.pdf.cell(0, 8, "AI Risk Assessment", ln=True)
        self.pdf.set_font("Helvetica", "", 10)
        risk = self.results["risk_score"]
        status = "HIGH RISK - Immediate Action Required" if risk > 0.7 else "Moderate Risk"
        self.pdf.set_fill_color(255, 240, 240)
        self.pdf.cell(0, 8, f"Transition Risk Score: {risk:.2f} ({status})", ln=True, fill=True)
        self.pdf.ln(5)

    def add_biomarkers(self):
        self.pdf.set_font("Helvetica", "B", 13)
        self.pdf.cell(0, 8, "Top Gradient-Driven Biomarkers", ln=True)
        data = self.results.get("top_biomarkers", [])
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.cell(45, 7, "Gene", border=1)
        self.pdf.cell(45, 7, "Gradient", border=1)
        self.pdf.cell(90, 7, "Role", border=1)
        self.pdf.ln()
        for item in data[:5]:
            self.pdf.cell(45, 7, item["gene"], border=1)
            self.pdf.cell(45, 7, f"{item['gradient']:.2f}", border=1)
            self.pdf.cell(90, 7, item["role"], border=1)
            self.pdf.ln()
        self.pdf.ln(8)

    def add_recommendation(self):
        self.pdf.set_font("Helvetica", "B", 13)
        self.pdf.cell(0, 8, "AI Recommendation", ln=True)
        self.pdf.set_font("Helvetica", "", 10)
        rec = self.results.get("recommendation", "Schedule colonoscopy within 3 months. Monitor ctDNA for KRAS mutations.")
        self.pdf.multi_cell(0, 6, rec)
        
    def save(self, filename="LOGMAC_Report.pdf"):
        self.add_header()
        self.add_risk_summary()
        self.add_biomarkers()
        self.add_recommendation()
        self.pdf.output(filename)
        return filename
from fpdf import FPDF

def save_report_as_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer Churn Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    
    for line in report_text.split('\n'):
        if line.startswith(("Prediction", "Top Drivers", "Recommendations")):
            pdf.set_font('', 'B', 12)  # Bold for headers
            pdf.cell(200, 10, txt=line, ln=1)
            pdf.set_font('', '', 12)
        else:
            pdf.cell(200, 8, txt=line, ln=1)
    
    pdf.output("report.pdf")
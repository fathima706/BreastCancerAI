from fpdf import FPDF
from datetime import datetime


def create_report(data):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        16
    )

    pdf.cell(
        0,
        10,
        "Breast Cancer AI Prediction Report",
        ln=True,
        align="C"
    )


    pdf.ln(10)


    pdf.set_font(
        "Arial",
        "",
        12
    )


    pdf.cell(
        0,
        10,
        f"Generated: {datetime.now()}",
        ln=True
    )


    pdf.ln(5)


    pdf.set_font(
        "Arial",
        "B",
        13
    )

    pdf.cell(
        0,
        10,
        "Patient Information",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        12
    )


    for key, value in data.items():

        pdf.cell(
            0,
            8,
            f"{key}: {value}",
            ln=True
        )


    pdf.ln(5)


    pdf.set_font(
        "Arial",
        "B",
        13
    )


    pdf.cell(
        0,
        10,
        "Clinical Prediction",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        12
    )


    pdf.cell(
        0,
        8,
        f"Result: {data['Prediction']}",
        ln=True
    )


    pdf.cell(
        0,
        8,
        f"Confidence: {data['Confidence']}%",
        ln=True
    )


    pdf.ln(10)


    pdf.multi_cell(
        0,
        8,
        "Disclaimer: This AI system is designed for educational and research purposes only. It does not replace professional medical diagnosis."
    )


    return bytes(pdf.output(dest="S"))
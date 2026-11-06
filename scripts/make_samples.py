"""Generate SYNTHETIC reports (fake patient, fake lab) for testing the app.

Usage:  python scripts/make_samples.py
Writes: sample_reports/sample_report_2026-03.pdf, sample_report_2026-09.pdf (lab reports)
        sample_mri_knee.pdf (a written MRI report) and sample_breast_biopsy.pdf (a pathology report)
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "sample_reports"

# (panel, test, unit, reference range, March value, September value)
TESTS = [
    ("Complete Blood Count", "Haemoglobin", "g/dL", "13.0 - 17.0", "11.8", "13.4"),
    ("Complete Blood Count", "Total WBC Count", "cells/cumm", "4000 - 11000", "7200", "6900"),
    ("Complete Blood Count", "Platelet Count", "lakh/cumm", "1.5 - 4.1", "2.6", "2.4"),
    ("Complete Blood Count", "MCV", "fL", "83 - 101", "76", "84"),
    ("Diabetes", "Fasting Blood Glucose", "mg/dL", "70 - 100", "112", "104"),
    ("Diabetes", "HbA1c", "%", "4.0 - 5.6", "6.1", "5.9"),
    ("Lipid Profile", "Total Cholesterol", "mg/dL", "< 200", "232", "238"),
    ("Lipid Profile", "LDL Cholesterol", "mg/dL", "< 100", "151", "162"),
    ("Lipid Profile", "HDL Cholesterol", "mg/dL", "> 40", "38", "41"),
    ("Lipid Profile", "Triglycerides", "mg/dL", "< 150", "198", "176"),
    ("Thyroid", "TSH", "µIU/mL", "0.27 - 4.20", "3.1", "2.9"),
    ("Liver Function", "SGPT (ALT)", "U/L", "< 41", "58", "36"),
    ("Kidney Function", "Serum Creatinine", "mg/dL", "0.7 - 1.3", "0.9", "1.0"),
    ("Vitamins", "Vitamin D (25-OH)", "ng/mL", "30 - 100", "14.2", "27.5"),
    ("Vitamins", "Vitamin B12", "pg/mL", "211 - 911", "420", "455"),
]


def flag(value: str, ref: str) -> str:
    v = float(value)
    ref = ref.replace(" ", "")
    if ref.startswith("<"):
        return "H" if v >= float(ref[1:]) else ""
    if ref.startswith(">"):
        return "L" if v <= float(ref[1:]) else ""
    lo, hi = (float(x) for x in ref.split("-"))
    return "L" if v < lo else "H" if v > hi else ""


def build(path: Path, date: str, col: int, sample_id: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm)
    story = [
        Paragraph("<b>DEMO DIAGNOSTICS LABORATORY</b> (sample data, not a real lab)", styles["Title"]),
        Paragraph("SYNTHETIC TEST REPORT FOR SOFTWARE TESTING ONLY", styles["Normal"]),
        Spacer(1, 6 * mm),
    ]
    header = Table(
        [
            ["Patient Name:", "Sample Patient", "Sample ID:", sample_id],
            ["Age / Sex:", "42 Y / Male", "Collected:", date],
            ["Referred By:", "Dr. Demo", "Reported:", date],
        ],
        colWidths=[28 * mm, 60 * mm, 25 * mm, 50 * mm],
    )
    header.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold")]))
    story += [header, Spacer(1, 6 * mm)]

    rows = [["Test", "Result", "Flag", "Unit", "Biological Ref. Range"]]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]
    panel = None
    for p, test, unit, ref, *values in TESTS:
        if p != panel:
            panel = p
            rows.append([p.upper(), "", "", "", ""])
            style += [("SPAN", (0, len(rows) - 1), (-1, len(rows) - 1)),
                      ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#e8eef5")),
                      ("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "Helvetica-Bold")]
        v = values[col]
        f = flag(v, ref)
        rows.append([test, v, f, unit, ref])
        if f:
            style.append(("FONTNAME", (1, len(rows) - 1), (2, len(rows) - 1), "Helvetica-Bold"))
    t = Table(rows, colWidths=[55 * mm, 22 * mm, 13 * mm, 25 * mm, 45 * mm])
    t.setStyle(TableStyle(style))
    story += [t, Spacer(1, 8 * mm),
              Paragraph("*** End of Report *** This is a synthetic document generated for testing.", styles["Italic"])]
    doc.build(story)


MRI_FINDINGS = [
    "Medial meniscus: Grade II intrasubstance signal in the posterior horn, not reaching the articular surface.",
    "Lateral meniscus: Normal in signal and morphology.",
    "Anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL): Intact.",
    "Medial and lateral collateral ligaments: Intact.",
    "Mild joint effusion is seen.",
    "Articular cartilage: Mild thinning over the medial femoral condyle.",
    "No bone marrow edema. No fracture.",
    "Patella and quadriceps / patellar tendons: Unremarkable.",
]
MRI_IMPRESSION = [
    "Grade II intrasubstance degeneration of the posterior horn of the medial meniscus. No meniscal tear.",
    "Mild joint effusion.",
    "Early chondral thinning of the medial femoral condyle.",
    "Cruciate and collateral ligaments intact.",
]


def build_mri(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm)
    story = [
        Paragraph("<b>DEMO IMAGING CENTRE</b> (sample data, not a real centre)", styles["Title"]),
        Paragraph("SYNTHETIC TEST REPORT FOR SOFTWARE TESTING ONLY", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph("<b>Patient:</b> Sample Patient &nbsp;&nbsp; <b>Age/Sex:</b> 42 Y / Male &nbsp;&nbsp; "
                  "<b>Date:</b> 20-Oct-2026 &nbsp;&nbsp; <b>Ref:</b> Dr. Demo", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph("<b>MRI LEFT KNEE (PLAIN)</b>", styles["Heading2"]),
        Paragraph("<b>Clinical history:</b> Pain in the left knee for 2 months, worse on climbing stairs.", styles["Normal"]),
        Paragraph("<b>Technique:</b> Multiplanar, multisequence MRI of the left knee without contrast.", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("<b>FINDINGS:</b>", styles["Normal"]),
        *[Paragraph(f"• {f}", styles["Normal"]) for f in MRI_FINDINGS],
        Spacer(1, 4 * mm),
        Paragraph("<b>IMPRESSION:</b>", styles["Normal"]),
        *[Paragraph(f"{i}. {f}", styles["Normal"]) for i, f in enumerate(MRI_IMPRESSION, 1)],
        Spacer(1, 4 * mm),
        Paragraph("Suggested clinical correlation.", styles["Normal"]),
        Spacer(1, 8 * mm),
        Paragraph("*** End of Report *** This is a synthetic document generated for testing.", styles["Italic"]),
    ]
    doc.build(story)


def build_biopsy(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm)
    rows = [
        ["Histological type", "Invasive ductal carcinoma, no special type (NST)"],
        ["Histological grade", "Grade 2 (Nottingham score 6/9: tubules 3, nuclei 2, mitoses 1)"],
        ["Tumour size", "2.4 cm in greatest dimension"],
        ["Margins", "All margins free of tumour; closest margin (superior) 3 mm"],
        ["Lymphovascular invasion", "Not identified"],
        ["Lymph nodes", "1 of 12 axillary lymph nodes shows metastatic carcinoma (1/12), size 4 mm"],
        ["Pathological stage (AJCC 8th ed.)", "pT2 pN1a"],
    ]
    ihc = [
        ["Estrogen receptor (ER)", "Positive, 90% of tumour cells, strong intensity (Allred score 8/8)"],
        ["Progesterone receptor (PR)", "Positive, 40% of tumour cells, moderate intensity (Allred score 6/8)"],
        ["HER2/neu", "Negative (IHC score 1+)"],
        ["Ki-67", "18%"],
    ]
    table_style = TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    def table(data):
        t = Table([[Paragraph(a, styles["Normal"]), Paragraph(b, styles["Normal"])] for a, b in data],
                  colWidths=[55 * mm, 115 * mm])
        t.setStyle(table_style)
        return t
    story = [
        Paragraph("<b>DEMO PATHOLOGY LABORATORY</b> (sample data, not a real lab)", styles["Title"]),
        Paragraph("SYNTHETIC TEST REPORT FOR SOFTWARE TESTING ONLY", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph("<b>Patient:</b> Sample Patient &nbsp;&nbsp; <b>Age/Sex:</b> 48 Y / Female &nbsp;&nbsp; "
                  "<b>Received:</b> 28-Oct-2026 &nbsp;&nbsp; <b>Ref:</b> Dr. Demo", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("<b>HISTOPATHOLOGY REPORT</b>", styles["Heading2"]),
        Paragraph("<b>Specimen:</b> Left breast, wide local excision with axillary lymph node dissection.", styles["Normal"]),
        Paragraph("<b>Clinical details:</b> Lump in left breast, upper outer quadrant.", styles["Normal"]),
        Spacer(1, 3 * mm),
        Paragraph("<b>Gross:</b> Received a fibro-fatty breast specimen measuring 7 x 5 x 4 cm with a firm grey-white "
                  "tumour measuring 2.4 x 2.0 x 1.8 cm. Twelve lymph nodes identified.", styles["Normal"]),
        Spacer(1, 3 * mm),
        Paragraph("<b>MICROSCOPY AND SYNOPTIC SUMMARY</b>", styles["Normal"]),
        table(rows),
        Spacer(1, 4 * mm),
        Paragraph("<b>IMMUNOHISTOCHEMISTRY</b>", styles["Normal"]),
        table(ihc),
        Spacer(1, 4 * mm),
        Paragraph("<b>FINAL DIAGNOSIS:</b> Left breast, wide local excision: Invasive ductal carcinoma (NST), "
                  "Grade 2, measuring 2.4 cm. Margins free. Metastatic carcinoma in 1 of 12 axillary lymph nodes. "
                  "pT2 pN1a. ER positive, PR positive, HER2 negative (1+). Ki-67 18%.", styles["Normal"]),
        Spacer(1, 3 * mm),
        Paragraph("Please correlate clinically. Discuss at multidisciplinary tumour board.", styles["Normal"]),
        Spacer(1, 8 * mm),
        Paragraph("*** End of Report *** This is a synthetic document generated for testing.", styles["Italic"]),
    ]
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build(OUT / "sample_report_2026-03.pdf", "12-Mar-2026", 0, "DEMO-0312")
    build(OUT / "sample_report_2026-09.pdf", "15-Sep-2026", 1, "DEMO-0915")
    build_mri(OUT / "sample_mri_knee.pdf")
    build_biopsy(OUT / "sample_breast_biopsy.pdf")
    print("Wrote", *sorted(p.name for p in OUT.glob("*.pdf")))

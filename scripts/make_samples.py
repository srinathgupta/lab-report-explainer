"""Generate two SYNTHETIC lab reports (fake patient, fake lab) for testing the app.

Usage:  python scripts/make_samples.py
Writes: sample_reports/sample_report_2026-03.pdf and sample_report_2026-09.pdf
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build(OUT / "sample_report_2026-03.pdf", "12-Mar-2026", 0, "DEMO-0312")
    build(OUT / "sample_report_2026-09.pdf", "15-Sep-2026", 1, "DEMO-0915")
    print("Wrote", *sorted(p.name for p in OUT.glob("*.pdf")))

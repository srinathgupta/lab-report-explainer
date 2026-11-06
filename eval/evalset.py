"""Generate a SYNTHETIC evaluation set: fake lab reports and fake cancer biopsy reports,
each with a ground-truth answer file. Nothing here comes from real patients.

    python -m eval.evalset --lab 20 --onco 10 --out eval_data

Every case is reproducible from its seed, so results can be compared across runs.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from medexplain.lab import computed_status

# name, unit, low, high (None = one-sided), decimals
LAB_TESTS = [
    ("Haemoglobin", "g/dL", 13.0, 17.0, 1),
    ("Total WBC Count", "cells/cumm", 4000, 11000, 0),
    ("Platelet Count", "lakh/cumm", 1.5, 4.1, 1),
    ("RBC Count", "million/cumm", 4.5, 5.5, 2),
    ("MCV", "fL", 83, 101, 0),
    ("Fasting Blood Glucose", "mg/dL", 70, 100, 0),
    ("HbA1c", "%", 4.0, 5.6, 1),
    ("Total Cholesterol", "mg/dL", None, 200, 0),
    ("LDL Cholesterol", "mg/dL", None, 100, 0),
    ("HDL Cholesterol", "mg/dL", 40, None, 0),
    ("Triglycerides", "mg/dL", None, 150, 0),
    ("TSH", "µIU/mL", 0.27, 4.20, 2),
    ("SGPT (ALT)", "U/L", None, 41, 0),
    ("SGOT (AST)", "U/L", None, 40, 0),
    ("Serum Creatinine", "mg/dL", 0.7, 1.3, 2),
    ("Blood Urea", "mg/dL", 15, 40, 0),
    ("Uric Acid", "mg/dL", 3.5, 7.2, 1),
    ("Vitamin D (25-OH)", "ng/mL", 30, 100, 1),
    ("Vitamin B12", "pg/mL", 211, 911, 0),
    ("Serum Sodium", "mmol/L", 136, 145, 0),
    ("Serum Potassium", "mmol/L", 3.5, 5.1, 1),
]


@dataclass
class LabTruth:
    name: str
    value: str
    unit: str
    reference_range: str
    status: str  # normal / high / low, per the printed range


@dataclass
class Case:
    case_id: str
    kind: str  # "lab" or "onco"
    pdf: str
    seed: int
    lab_tests: list[LabTruth] = field(default_factory=list)
    onco: dict = field(default_factory=dict)
    notes: str = ""


def _fmt(v: float, d: int) -> str:
    return f"{v:.{d}f}"


def _range_text(lo, hi, d, style: int) -> str:
    if lo is None:
        return f"< {_fmt(hi, d)}" if style else f"<{_fmt(hi, d)}"
    if hi is None:
        return f"> {_fmt(lo, d)}" if style else f">{_fmt(lo, d)}"
    sep = [" - ", "-", " – "][style % 3]
    return f"{_fmt(lo, d)}{sep}{_fmt(hi, d)}"


def _sample_value(rng: random.Random, lo, hi, d) -> tuple[float, str]:
    """Mostly normal, some clearly abnormal, a few exactly on the boundary (which counts as normal)."""
    r = rng.random()
    lo_eff = lo if lo is not None else hi * 0.5
    hi_eff = hi if hi is not None else lo * 2.0
    if r < 0.10 and lo is not None and hi is not None:
        # exactly on a two-sided limit counts as normal; skipped for "< 200"-style ranges, where it's ambiguous
        return rng.choice([lo, hi]), "boundary"
    if r < 0.60:
        return rng.uniform(lo_eff, hi_eff), "normal"
    if r < 0.80 and lo is not None:
        return lo * rng.uniform(0.6, 0.95), "low"
    return hi_eff * rng.uniform(1.05, 1.5), "high"


def make_lab_case(i: int, seed: int, out: Path) -> Case:
    rng = random.Random(seed)
    tests = rng.sample(LAB_TESTS, rng.randint(8, 14))
    style = rng.randint(0, 2)
    show_flag = rng.random() < 0.5
    truths: list[LabTruth] = []
    for name, unit, lo, hi, d in tests:
        v, _ = _sample_value(rng, lo, hi, d)
        value = _fmt(v, d)
        ref = _range_text(lo, hi, d, style)
        truths.append(LabTruth(name, value, unit, ref, computed_status(value, ref) or "unknown"))

    pdf = out / f"lab_{i:03d}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm)
    header = ["Test", "Result", "Unit", ["Biological Ref. Range", "Reference Interval", "Normal Range"][style]]
    if show_flag:
        header.insert(2, "Flag")
    rows = [header]
    for t in truths:
        row = [t.name, t.value, t.unit, t.reference_range]
        if show_flag:
            row.insert(2, {"high": "H", "low": "L"}.get(t.status, ""))
        rows.append(row)
    table = Table(rows)
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(["#1f4e79", "#2e7d32", "#555555"][style])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    lab = ["DEMO DIAGNOSTICS", "SAMPLE PATH LABS", "TEST HEALTH CENTRE"][style]
    doc.build([
        Paragraph(f"<b>{lab}</b> (synthetic evaluation report, not a real lab)", styles["Title"]),
        Paragraph(f"Patient: Eval Patient {i} &nbsp; Age/Sex: {rng.randint(25, 75)} Y / {rng.choice(['Male', 'Female'])} "
                  f"&nbsp; Date: {rng.randint(1, 28):02d}-Oct-2026", styles["Normal"]),
        Spacer(1, 5 * mm), table,
    ])
    return Case(f"lab_{i:03d}", "lab", pdf.name, seed, lab_tests=truths)


ONCO_SITES = {
    "breast": {
        "diagnosis": "Invasive ductal carcinoma, no special type (NST)",
        "organ": "Breast",
        "markers": lambda rng: {
            "ER": rng.choice(["Positive (90%)", "Positive (40%)", "Negative (0%)"]),
            "PR": rng.choice(["Positive (60%)", "Negative (0%)"]),
            "HER2": rng.choice(["Negative (IHC 0)", "Negative (IHC 1+)", "Equivocal (IHC 2+)", "Positive (IHC 3+)"]),
            "Ki-67": f"{rng.choice([8, 15, 25, 40, 60])}%",
        },
    },
    "colon": {
        "diagnosis": "Adenocarcinoma of the colon",
        "organ": "Sigmoid colon",
        "markers": lambda rng: {
            "MMR": rng.choice(["Proficient (MLH1, MSH2, MSH6, PMS2 retained)", "Deficient (loss of MLH1 and PMS2)"]),
            "KRAS": rng.choice(["No mutation detected", "Mutation detected (G12D)"]),
        },
    },
    "lung": {
        "diagnosis": "Adenocarcinoma of the lung",
        "organ": "Right upper lobe of lung",
        "markers": lambda rng: {
            "PD-L1": f"TPS {rng.choice([0, 5, 30, 70])}%",
            "EGFR": rng.choice(["No mutation detected", "Exon 19 deletion detected"]),
            "ALK": rng.choice(["Negative", "Positive (rearrangement detected)"]),
        },
    },
}
GRADE_TEXT = {1: ["Grade 1", "Well differentiated"], 2: ["Grade 2", "Moderately differentiated"],
              3: ["Grade 3", "Poorly differentiated"]}


def make_onco_case(i: int, seed: int, out: Path) -> Case:
    rng = random.Random(seed)
    site = list(ONCO_SITES)[i % len(ONCO_SITES)]  # rotate evenly through sites
    spec = ONCO_SITES[site]
    t = rng.choice(["pT1b", "pT1c", "pT2", "pT3", "pT4a"])
    n = rng.choice(["pN0", "pN1a", "pN2"])
    tnm = f"{t} {n}"
    grade = rng.randint(1, 3)
    grade_text = rng.choice(GRADE_TEXT[grade])
    markers = spec["markers"](rng)
    nodes = {"pN0": "0 of 14", "pN1a": "2 of 14", "pN2": "5 of 15"}[n]
    margins = rng.choice(["All margins free of tumour (closest 4 mm)", "Tumour present at the deep margin"])

    pdf = out / f"onco_{i:03d}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm)
    rows = [["Histological type", spec["diagnosis"]], ["Histological grade", grade_text],
            ["Tumour size", f"{rng.uniform(0.8, 6.5):.1f} cm"], ["Margins", margins],
            ["Lymph nodes", f"{nodes} lymph nodes involved"], ["Pathological stage", tnm]]
    rows += [[k, v] for k, v in markers.items()]
    table = Table([[Paragraph(a, styles["Normal"]), Paragraph(b, styles["Normal"])] for a, b in rows],
                  colWidths=[55 * mm, 115 * mm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    doc.build([
        Paragraph("<b>DEMO PATHOLOGY LAB</b> (synthetic evaluation report, not a real lab)", styles["Title"]),
        Paragraph(f"<b>HISTOPATHOLOGY REPORT</b> · Specimen: {spec['organ']}, resection", styles["Normal"]),
        Spacer(1, 4 * mm), table, Spacer(1, 4 * mm),
        Paragraph(f"<b>FINAL DIAGNOSIS:</b> {spec['diagnosis']}, {grade_text.lower()}. {tnm}.", styles["Normal"]),
    ])
    truth = {"site": site, "tnm": tnm, "grade": grade, "markers": list(markers)}
    return Case(f"onco_{i:03d}", "onco", pdf.name, seed, onco=truth)


def build(out: Path, n_lab: int, n_onco: int, base_seed: int = 2026) -> list[Case]:
    out.mkdir(parents=True, exist_ok=True)
    cases = [make_lab_case(i, base_seed + i, out) for i in range(n_lab)]
    cases += [make_onco_case(i, base_seed + 1000 + i, out) for i in range(n_onco)]
    (out / "truth.json").write_text(json.dumps([asdict(c) for c in cases], indent=1, ensure_ascii=False))
    return cases


def load(out: Path) -> list[Case]:
    raw = json.loads((out / "truth.json").read_text())
    return [Case(**{**c, "lab_tests": [LabTruth(**t) for t in c["lab_tests"]]}) for c in raw]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", type=int, default=20)
    ap.add_argument("--onco", type=int, default=10)
    ap.add_argument("--out", default="eval_data")
    a = ap.parse_args()
    print(f"Wrote {len(build(Path(a.out), a.lab, a.onco))} cases to {a.out}/")

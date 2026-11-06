"""Side-effect tracker for people in cancer treatment.

Everything here is plain code: no AI is involved, so the warning signs are always the same
fixed rules. The rules follow common chemotherapy patient-education advice and MUST be
reviewed by a doctor before real-world use. Patients should always follow their own care
team's instructions, which take priority over this app.

Nothing is stored on a server: the log lives in the browser session and the patient
downloads it as a JSON file, then loads it again next time.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .core import SummarizerError

# ---------------------------------------------------------------------------
# What we track
# ---------------------------------------------------------------------------

SCALE_LABELS = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Very severe"}

# 0-4 scale symptoms: key -> (label, short hint shown in the form)
SCALE_SYMPTOMS = {
    "nausea": ("Nausea (feeling sick)", "Mild = some loss of appetite; Severe = can hardly eat or drink"),
    "tiredness": ("Tiredness", "Severe = can't do daily activities like bathing or dressing"),
    "mouth_sores": ("Mouth sores", "Severe = can't eat or drink because of pain"),
    "numbness": ("Numbness or tingling", "In hands or feet; Severe = can't button clothes or walk safely"),
    "appetite_loss": ("Loss of appetite", ""),
    "pain": ("Pain", "Anywhere in the body"),
    "breathlessness": ("Shortness of breath", "Severe = breathless at rest"),
    "skin": ("Skin rash or redness", ""),
}
# counts in the last 24 hours
COUNT_SYMPTOMS = {
    "vomiting_times": "Times vomited (last 24 hours)",
    "loose_stools": "Extra loose stools compared with your normal (last 24 hours)",
}
# yes / no warning signs
FLAG_SYMPTOMS = {
    "cannot_drink": "Unable to keep fluids down",
    "bleeding": "Unusual bleeding or bruising",
    "chest_pain": "Chest pain",
    "confusion": "New confusion or unusual drowsiness",
}

FEVER_C = 38.0
WATCH_TEMP_C = 37.5
EMERGENCY_NUMBERS = "112 (emergency) or 108 (ambulance)"
MAX_ENTRIES = 400
MAX_NOTE_CHARS = 300
MAX_LOG_BYTES = 512 * 1024


class Entry(BaseModel):
    day: date
    temperature_c: Optional[float] = Field(default=None, ge=30, le=45)
    scores: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    flags: dict[str, bool] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=MAX_NOTE_CHARS)

    @field_validator("scores")
    @classmethod
    def _scores(cls, v: dict[str, int]) -> dict[str, int]:
        return {k: max(0, min(4, int(x))) for k, x in v.items() if k in SCALE_SYMPTOMS}

    @field_validator("counts")
    @classmethod
    def _counts(cls, v: dict[str, int]) -> dict[str, int]:
        return {k: max(0, min(50, int(x))) for k, x in v.items() if k in COUNT_SYMPTOMS}

    @field_validator("flags")
    @classmethod
    def _flags(cls, v: dict[str, bool]) -> dict[str, bool]:
        return {k: bool(x) for k, x in v.items() if k in FLAG_SYMPTOMS}


class Log(BaseModel):
    app: Literal["lab-report-explainer-tracker"] = "lab-report-explainer-tracker"
    version: int = 1
    entries: list[Entry] = Field(default_factory=list, max_length=MAX_ENTRIES)

    def sorted_entries(self) -> list[Entry]:
        return sorted(self.entries, key=lambda e: e.day)


def add_entry(log: Log, entry: Entry) -> Log:
    """Add an entry, replacing any existing entry for the same day."""
    entries = [e for e in log.entries if e.day != entry.day] + [entry]
    if len(entries) > MAX_ENTRIES:
        raise SummarizerError(f"The log can hold at most {MAX_ENTRIES} days. Please start a new log.")
    return Log(entries=sorted(entries, key=lambda e: e.day))


def remove_entry(log: Log, day: date) -> Log:
    return Log(entries=[e for e in log.entries if e.day != day])


def dump_log(log: Log) -> bytes:
    return log.model_dump_json(indent=2).encode("utf-8")


def load_log(data: bytes) -> Log:
    if len(data) > MAX_LOG_BYTES:
        raise SummarizerError("This file is too large to be a side-effect log.")
    try:
        return Log.model_validate_json(data)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        raise SummarizerError(
            "This file isn't a side-effect log from this app. Please choose the file you downloaded earlier."
        ) from e


# ---------------------------------------------------------------------------
# Warning rules (fixed, human-reviewable)
# ---------------------------------------------------------------------------

Level = Literal["emergency", "urgent", "soon", "watch"]
LEVEL_ORDER = {"emergency": 0, "urgent": 1, "soon": 2, "watch": 3}


@dataclass(frozen=True)
class Alert:
    level: Level
    message: str
    rule: str  # short id, useful in tests
    title: str = ""  # short label for the doctor summary


def check_entry(entry: Entry) -> list[Alert]:
    """Warning signs for a single day."""
    alerts: list[Alert] = []
    s, c, f, t = entry.scores, entry.counts, entry.flags, entry.temperature_c

    if f.get("chest_pain") or s.get("breathlessness", 0) >= 3:
        alerts.append(Alert("emergency", "Chest pain or severe breathlessness: call "
                            f"{EMERGENCY_NUMBERS} now, then tell your cancer care team.", "breathing", "Chest pain or severe breathlessness"))
    if t is not None and t >= FEVER_C:
        alerts.append(Alert("urgent", f"Temperature {t:.1f}°C. A temperature of {FEVER_C:.0f}°C or higher during "
                            "cancer treatment can be a sign of a serious infection. Contact your cancer care team "
                            "or their emergency helpline now, any time of day or night.", "fever", f"Temperature {t:.1f}°C"))
    elif t is not None and t >= WATCH_TEMP_C:
        alerts.append(Alert("watch", f"Temperature {t:.1f}°C. Check it again in an hour. If it reaches "
                            f"{FEVER_C:.0f}°C or you feel shivery or unwell, contact your care team now.", "warm", f"Temperature {t:.1f}°C"))
    if f.get("bleeding"):
        alerts.append(Alert("urgent", "Unusual bleeding or bruising: contact your cancer care team now.", "bleeding", "Unusual bleeding or bruising"))
    if f.get("confusion"):
        alerts.append(Alert("urgent", "New confusion or unusual drowsiness: contact your cancer care team now.", "confusion", "New confusion or drowsiness"))
    if f.get("cannot_drink") or c.get("vomiting_times", 0) >= 3:
        alerts.append(Alert("urgent", "Vomiting 3 or more times, or unable to keep fluids down: contact your "
                            "cancer care team now. Dehydration can become serious quickly.", "vomiting", f"Vomited {c.get('vomiting_times', 0)} time(s) / unable to keep fluids down" if f.get("cannot_drink") else f"Vomited {c.get('vomiting_times', 0)} times"))
    if c.get("loose_stools", 0) >= 4:
        alerts.append(Alert("urgent", "4 or more extra loose stools in a day: contact your cancer care team now.",
                            "diarrhoea", f"{c.get('loose_stools', 0)} extra loose stools"))

    for key, score in s.items():
        label = SCALE_SYMPTOMS[key][0]
        if key == "breathlessness" and score >= 3:
            continue  # already an emergency above
        if score == 4:
            alerts.append(Alert("urgent", f"{label} is very severe: contact your cancer care team now.", f"{key}_4", f"{label}: very severe"))
        elif score == 3:
            alerts.append(Alert("soon", f"{label} is severe: contact your cancer care team today.", f"{key}_3", f"{label}: severe"))
    return sorted(alerts, key=lambda a: LEVEL_ORDER[a.level])


def check_trends(log: Log, days: int = 3) -> list[Alert]:
    """Symptoms that have stayed moderate or worse for `days` logged days in a row."""
    entries = log.sorted_entries()
    alerts = []
    for key, (label, _) in SCALE_SYMPTOMS.items():
        run = 0
        prev: Optional[date] = None
        for e in entries:
            consecutive = prev is not None and e.day - prev == timedelta(days=1)
            if e.scores.get(key, 0) >= 2:
                run = run + 1 if consecutive or run == 0 else 1
            else:
                run = 0
            prev = e.day
        if run >= days:
            alerts.append(Alert("soon", f"{label} has been moderate or worse for {run} days in a row. "
                                "Let your cancer care team know; there may be ways to help.", f"{key}_trend"))
    return alerts


def latest_alerts(log: Log) -> list[Alert]:
    entries = log.sorted_entries()
    if not entries:
        return []
    return sorted(check_entry(entries[-1]) + check_trends(log), key=lambda a: LEVEL_ORDER[a.level])


# ---------------------------------------------------------------------------
# Summary for the oncologist
# ---------------------------------------------------------------------------


def summary_rows(log: Log) -> dict:
    """Plain numbers for the doctor summary (also used in tests)."""
    entries = log.sorted_entries()
    temps = [(e.day, e.temperature_c) for e in entries if e.temperature_c is not None]
    symptoms = []
    for key, (label, _) in SCALE_SYMPTOMS.items():
        vals = [(e.day, e.scores.get(key, 0)) for e in entries]
        worst_day, worst = max(vals, key=lambda x: (x[1], x[0])) if vals else (None, 0)
        days_moderate = sum(1 for _, v in vals if v >= 2)
        latest = vals[-1][1] if vals else 0
        if worst > 0:
            symptoms.append({"label": label, "worst": worst, "worst_day": worst_day,
                             "days_moderate": days_moderate, "latest": latest})
    events = [(e.day, a) for e in entries for a in check_entry(e) if a.level in ("emergency", "urgent")]
    return {
        "first": entries[0].day if entries else None,
        "last": entries[-1].day if entries else None,
        "days_logged": len(entries),
        "max_temp": max(temps, key=lambda x: x[1]) if temps else None,
        "fever_days": [d for d, t in temps if t >= FEVER_C],
        "max_vomiting": max((e.counts.get("vomiting_times", 0) for e in entries), default=0),
        "max_loose_stools": max((e.counts.get("loose_stools", 0) for e in entries), default=0),
        "symptoms": symptoms,
        "events": events,
        "notes": [(e.day, e.notes) for e in entries if e.notes.strip()],
    }


def doctor_summary_pdf(log: Log) -> bytes:
    """A one-page (or short) PDF the patient can show at their next appointment."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    r = summary_rows(log)
    if not r["days_logged"]:
        raise SummarizerError("The log is empty. Add at least one day before making a summary.")
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm)
    fmt = lambda d: d.strftime("%d %b %Y")  # noqa: E731
    story = [
        Paragraph("Side-effect summary for my care team", styles["Title"]),
        Paragraph(f"{fmt(r['first'])} to {fmt(r['last'])} · {r['days_logged']} day(s) logged by the patient", styles["Normal"]),
        Spacer(1, 4 * mm),
    ]
    facts = []
    if r["max_temp"]:
        facts.append(f"Highest temperature: <b>{r['max_temp'][1]:.1f}°C</b> on {fmt(r['max_temp'][0])}")
    if r["fever_days"]:
        facts.append(f"Days with temperature {FEVER_C:.0f}°C or higher: <b>{', '.join(fmt(d) for d in r['fever_days'])}</b>")
    if r["max_vomiting"]:
        facts.append(f"Most vomiting in one day: <b>{r['max_vomiting']} time(s)</b>")
    if r["max_loose_stools"]:
        facts.append(f"Most extra loose stools in one day: <b>{r['max_loose_stools']}</b>")
    for line in facts or ["No temperature, vomiting or diarrhoea problems recorded."]:
        story.append(Paragraph("• " + line, styles["Normal"]))
    story.append(Spacer(1, 4 * mm))

    if r["symptoms"]:
        story.append(Paragraph("<b>Symptoms (0 none – 4 very severe)</b>", styles["Normal"]))
        rows = [["Symptom", "Worst", "Worst on", "Days moderate+", "Latest"]] + [
            [s["label"], f"{s['worst']} {SCALE_LABELS[s['worst']]}", fmt(s["worst_day"]), str(s["days_moderate"]),
             f"{s['latest']} {SCALE_LABELS[s['latest']]}"] for s in r["symptoms"]
        ]
        table = Table(rows, colWidths=[52 * mm, 32 * mm, 30 * mm, 28 * mm, 32 * mm])
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story += [table, Spacer(1, 4 * mm)]

    if r["events"]:
        story.append(Paragraph("<b>Warning signs flagged by the app</b>", styles["Normal"]))
        for d, a in r["events"]:
            story.append(Paragraph(f"• {fmt(d)}: {escape(a.title or a.rule)}", styles["Normal"]))
        story.append(Spacer(1, 4 * mm))

    if r["notes"]:
        story.append(Paragraph("<b>Patient's notes</b>", styles["Normal"]))
        for d, n in r["notes"]:
            story.append(Paragraph(f"• {fmt(d)}: {escape(n)}", styles["Normal"]))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "<i>Self-reported by the patient using Lab Report Explainer (an educational tool, not a medical device). "
        "Warning rules are general guidance; the care team's own instructions take priority.</i>", styles["Normal"]))
    doc.build(story)
    return buf.getvalue()

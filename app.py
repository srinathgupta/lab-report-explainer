"""Lab Report Explainer — Streamlit UI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from summarizer import (
    DEFAULT_MODEL,
    LANGUAGES,
    MAX_FOLLOWUPS,
    MAX_NOTE_FILES,
    MAX_QUESTION_CHARS,
    UPLOAD_TYPES,
    ComparisonSummary,
    NoteSummary,
    ReportSummary,
    ScanSummary,
    SummarizerError,
    ask_followup,
    compare_reports,
    explain_note,
    explain_scan,
    get_api_key,
    summarize_report,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CONFIG_NAMES = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "MAX_PDF_PAGES",
    "SESSION_LIMIT_PER_HOUR",
    "GLOBAL_LIMIT_PER_HOUR",
    "DEMO_MODE",
)


def load_streamlit_secrets() -> None:
    """On Streamlit Community Cloud the key lives in the app's Secrets; expose it as an env var."""
    for name in CONFIG_NAMES:
        if os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets.toml locally
            return
        if value:
            os.environ[name] = str(value)


load_streamlit_secrets()

STATUS_LABELS = {
    "normal": "🟢 Normal",
    "high": "🔴 High",
    "low": "🔵 Low",
    "borderline": "🟡 Borderline",
    "abnormal": "🔴 Abnormal",
    "unknown": "⚪ No range",
}
OUT_OF_RANGE = {"high", "low", "abnormal", "borderline"}

CHANGE_LABELS = {
    "improved": "✅ Improved",
    "worsened": "⚠️ Worsened",
    "stable": "➖ Stable",
    "changed": "🔄 Changed",
    "new_test": "🆕 New test",
    "not_retested": "— Not retested",
    "unclear": "❔ Unclear",
}

SAMPLES_DIR = Path(__file__).resolve().parent / "sample_reports"
SAMPLE_EARLIER = SAMPLES_DIR / "sample_report_2026-03.pdf"
SAMPLE_LATER = SAMPLES_DIR / "sample_report_2026-09.pdf"
SAMPLE_SCAN = SAMPLES_DIR / "sample_mri_knee.pdf"
DEMO_BANNER = (
    "**Demo app.** Please don't upload real medical reports or prescriptions. Use the sample buttons "
    "or a made-up document. Files are processed by Google Gemini's free tier, which may use them to improve "
    "Google's products."
)


def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


DISCLAIMER = (
    "This tool explains medical reports in simple words. It is **not medical advice** and can make mistakes. "
    "Always discuss your results with a doctor before making any health decision."
)


# ---------------------------------------------------------------------------
# Usage limits (protect the API key from abuse and runaway costs)
# ---------------------------------------------------------------------------

WINDOW_S = 3600


def _limit(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@st.cache_resource
def _global_usage() -> tuple[deque, threading.Lock]:
    """Shared by every visitor of this app process."""
    return deque(), threading.Lock()


def _prune(times, now: float) -> None:
    while times and now - times[0] > WINDOW_S:
        times.popleft()


def try_use_quota(now: float | None = None) -> str | None:
    """Record one request. Returns an error message if a limit is reached, else None."""
    now = time.time() if now is None else now
    per_session = _limit("SESSION_LIMIT_PER_HOUR", 10)
    per_app = _limit("GLOBAL_LIMIT_PER_HOUR", 60)

    session_times = deque(st.session_state.get("usage_times", []))
    _prune(session_times, now)
    if len(session_times) >= per_session:
        wait = math.ceil((WINDOW_S - (now - session_times[0])) / 60)
        return f"You've reached the limit of {per_session} reports per hour. Please try again in about {wait} minutes."

    times, lock = _global_usage()
    with lock:
        _prune(times, now)
        if len(times) >= per_app:
            return "The app is getting a lot of requests right now. Please try again later."
        times.append(now)

    session_times.append(now)
    st.session_state["usage_times"] = list(session_times)
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_summary(summary: ReportSummary, corrected: list[str] | None = None) -> None:
    corrected = set(corrected or [])
    if not summary.is_lab_report:
        st.warning("This doesn't look like a medical lab report. Please upload a lab test report PDF.")
        return

    meta = " · ".join(x for x in [summary.lab_name, summary.report_date] if x)
    if meta:
        st.caption(meta)

    for flag in summary.urgent_flags:
        st.error(f"**Contact your doctor soon:** {flag}", icon="🚨")

    st.subheader("Summary")
    st.write(summary.overall_summary)

    flagged = [t for t in summary.tests if t.status in OUT_OF_RANGE]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tests found", len(summary.tests))
    c2.metric("In range", sum(t.status == "normal" for t in summary.tests))
    c3.metric("Out of range", len(flagged))

    if summary.key_findings:
        st.subheader("Key findings")
        st.markdown("\n".join(f"- {f}" for f in summary.key_findings))

    if summary.tests:
        st.subheader("All results")
        only_flagged = st.toggle("Show only out-of-range results", value=False, key="only_flagged")
        rows = flagged if only_flagged else summary.tests
        df = pd.DataFrame(
            [
                {
                    "Test": t.name,
                    "Result": f"{t.value} {t.unit}".strip(),
                    "Reference range": t.reference_range or "—",
                    "Status": STATUS_LABELS.get(t.status, t.status),
                }
                for t in rows
            ]
        )
        st.dataframe(df, hide_index=True)

        st.subheader("What each result means")
        # Out-of-range results first, then grouped by panel.
        ordered = sorted(rows, key=lambda t: (t.status not in OUT_OF_RANGE, t.category))
        for t in ordered:
            label = f"{STATUS_LABELS.get(t.status, t.status)} · {t.name}: {t.value} {t.unit}".strip()
            with st.expander(label, expanded=t.status in OUT_OF_RANGE):
                st.write(t.explanation)
                if t.reference_range:
                    st.caption(f"Reference range: {t.reference_range} · {t.category}")
                if t.name in corrected:
                    st.caption("✔ Status set by the app's own check against the printed range.")

    if summary.questions_for_doctor:
        st.subheader("Questions to ask your doctor")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(summary.questions_for_doctor, 1)))

    if summary.notes:
        st.info(summary.notes, icon="📝")

    if corrected:
        st.caption(
            f"An automatic check against the printed reference ranges corrected the status of: {', '.join(sorted(corrected))}. "
            "The written explanation for these may not match the corrected status."
        )


def render_comparison(cmp: ComparisonSummary) -> None:
    if not cmp.both_lab_reports:
        st.warning("At least one of these files doesn't look like a medical lab report.")
        return
    if not cmp.same_person_likely:
        st.warning("These reports may belong to different people. Check that you uploaded the right files.")

    dates = " → ".join(x for x in [cmp.earlier_report_date, cmp.later_report_date] if x)
    if dates:
        st.caption(dates)

    for flag in cmp.urgent_flags:
        st.error(f"**Contact your doctor soon:** {flag}", icon="🚨")

    st.subheader("What changed")
    st.write(cmp.overall_summary)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Better**")
        st.markdown("\n".join(f"- {x}" for x in cmp.improvements) or "_Nothing clearly improved._")
    with col_b:
        st.markdown("**⚠️ Needs attention**")
        st.markdown("\n".join(f"- {x}" for x in cmp.concerns) or "_No new concerns._")

    if cmp.trends:
        st.subheader("Side-by-side")
        df = pd.DataFrame(
            [
                {
                    "Test": t.name,
                    "Earlier": f"{t.earlier_value} {t.unit}".strip() if t.earlier_value else "—",
                    "Later": f"{t.later_value} {t.unit}".strip() if t.later_value else "—",
                    "Reference range": t.reference_range or "—",
                    "Change": CHANGE_LABELS.get(t.change, t.change),
                }
                for t in cmp.trends
            ]
        )
        st.dataframe(df, hide_index=True)

        with st.expander("Explanations"):
            for t in cmp.trends:
                st.markdown(f"**{t.name}** — {CHANGE_LABELS.get(t.change, t.change)}  \n{t.explanation}")

    if cmp.questions_for_doctor:
        st.subheader("Questions to ask your doctor")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(cmp.questions_for_doctor, 1)))

    if cmp.notes:
        st.info(cmp.notes, icon="📝")


CONFIDENCE_LABELS = {
    "high": "🟢 Clearly read",
    "medium": "🟡 Please confirm",
    "low": "🔴 Unclear, ask your pharmacist",
}
LEGIBILITY_WARNINGS = {
    "partly_unclear": "Parts of this note were hard to read. Items marked 🟡 or 🔴 may be wrong.",
    "mostly_unclear": "Most of this note was hard to read, so this explanation is unreliable. "
    "Please ask your doctor or pharmacist to read it with you.",
}
NOTE_WARNING = (
    "**Handwriting is easy to misread, for people and for AI.** Never start, stop or change a medicine "
    "based on this app. Always confirm every medicine name and dose with your pharmacist or doctor."
)


def render_items(title: str, items) -> None:
    if not items:
        return
    st.subheader(title)
    for item in items:
        badge = "" if item.confidence == "high" else f" · {CONFIDENCE_LABELS[item.confidence]}"
        st.markdown(f"**{item.as_written}**{badge}  \n{item.plain_explanation}")


def render_note(note: NoteSummary, codes: dict) -> None:
    if not note.is_medical_note:
        st.warning("This doesn't look like a prescription or doctor's note. Please upload a clear photo of one.")
        return

    if note.legibility in LEGIBILITY_WARNINGS:
        (st.error if note.legibility == "mostly_unclear" else st.warning)(LEGIBILITY_WARNINGS[note.legibility])
    kind = note.document_type.replace("_", " ").capitalize()
    st.caption(" · ".join(x for x in [kind, note.date] if x))

    st.subheader("Summary")
    st.write(note.overall_summary)

    render_items("What the doctor noted", note.findings)

    if note.medicines:
        st.subheader("Medicines")
        st.caption("Confidence is the AI's own rating and can be wrong. Check every medicine with your pharmacist.")
        for i, med in enumerate(note.medicines):
            title = f"{CONFIDENCE_LABELS[med.confidence]} · {med.name_as_written}"
            if med.strength:
                title += f" {med.strength}"
            with st.expander(title, expanded=True):
                st.markdown(f"**Written as:** {med.name_as_written} {med.strength} {med.form}".rstrip())
                if med.likely_medicine:
                    suffix = "" if med.confidence == "high" else " (please confirm)"
                    st.markdown(f"**Likely medicine:** {med.likely_medicine}{suffix}")
                if med.dose_as_written:
                    st.markdown(f"**How to take (as written):** `{med.dose_as_written}`")
                if med.dose_in_plain_words:
                    st.markdown(f"**In plain words:** {med.dose_in_plain_words}")
                if med.duration:
                    st.markdown(f"**For how long:** {med.duration}")
                for code, meaning in codes.get(str(i), []):
                    st.caption(f"`{code}` means: {meaning} (from a standard abbreviation list)")
                if med.usual_purpose:
                    st.markdown(f"**Commonly used for:** {med.usual_purpose}")
                if med.unclear_parts:
                    st.warning(f"Couldn't read clearly: {med.unclear_parts}")

    render_items("Tests the doctor asked for", note.tests_ordered)
    render_items("Other advice", note.advice)

    if note.follow_up:
        st.subheader("Follow-up")
        st.write(note.follow_up)

    if note.could_not_read:
        st.subheader("Couldn't read")
        st.markdown("\n".join(f"- {x}" for x in note.could_not_read))

    if note.questions_for_doctor_or_pharmacist:
        st.subheader("Questions to ask your doctor or pharmacist")
        st.markdown(
            "\n".join(f"{i}. {q}" for i, q in enumerate(note.questions_for_doctor_or_pharmacist, 1))
        )

    st.info(NOTE_WARNING, icon="💊")


SCAN_LABELS = {
    "xray": "X-ray",
    "ct": "CT scan",
    "mri": "MRI",
    "ultrasound": "Ultrasound",
    "ecg": "ECG",
    "echo": "Echocardiogram",
    "pathology": "Pathology / biopsy",
    "other": "Report",
}
FINDING_LABELS = {"normal": "🟢 Normal", "needs_attention": "🟠 Needs attention", "unclear": "⚪ Unclear"}
SCAN_WARNING = (
    "This explains the **written report** from your radiologist or lab. It does not look at the scan images "
    "themselves and is not a diagnosis. Discuss the report with the doctor who ordered the test."
)
SCAN_IMAGE_MESSAGE = (
    "This looks like the scan image itself, not the written report. This app only explains the written report: "
    "reading X-ray, CT or MRI images needs a trained radiologist. Please upload the report that came with your scan."
)


def render_findings(title: str, items, show_status: bool = True) -> None:
    if not items:
        return
    st.subheader(title)
    for f in sorted(items, key=lambda f: f.status != "needs_attention") if show_status else items:
        badge = f"{FINDING_LABELS[f.status]} · " if show_status else "📌 "
        st.markdown(f"{badge}**{f.as_written}**  \n{f.plain_explanation}")


def render_scan(scan: ScanSummary) -> None:
    if scan.document_kind == "scan_image":
        st.warning(SCAN_IMAGE_MESSAGE, icon="🩻")
        return
    if scan.document_kind == "other":
        st.warning("This doesn't look like a scan or test report. Please upload the written report.")
        return

    st.caption(" · ".join(x for x in [SCAN_LABELS[scan.report_type], scan.body_part, scan.date] if x))
    for flag in scan.urgent_flags:
        st.error(f"**Contact your doctor soon:** {flag}", icon="🚨")
    if scan.reason_for_test:
        st.markdown(f"**Why the test was done:** {scan.reason_for_test}")

    st.subheader("Summary")
    st.write(scan.overall_summary)

    all_findings = scan.findings
    if all_findings:
        c1, c2 = st.columns(2)
        c1.metric("Normal findings", sum(f.status == "normal" for f in all_findings))
        c2.metric("Need attention", sum(f.status == "needs_attention" for f in all_findings))

    render_findings("Main conclusion", scan.main_conclusion)
    render_findings("All findings", scan.findings)

    if scan.terms:
        st.subheader("Medical words explained")
        st.dataframe(
            pd.DataFrame([{"Word": x.term, "Meaning": x.meaning} for x in scan.terms]), hide_index=True
        )

    render_findings("What the report recommends", scan.recommendations, show_status=False)

    if scan.questions_for_doctor:
        st.subheader("Questions to ask your doctor")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(scan.questions_for_doctor, 1)))
    if scan.notes:
        st.info(scan.notes, icon="📝")
    st.info(SCAN_WARNING, icon="🩻")


FOLLOWUP_PLACEHOLDERS = {
    "single_result": "e.g. What can help bring my LDL cholesterol down?",
    "compare_result": "e.g. Why might my vitamin D have improved?",
    "scan_result": "e.g. What does 'joint effusion' mean?",
    "note_result": "e.g. What is a CBC test?",
}
EMERGENCY_MESSAGE = (
    "What you describe may need prompt medical attention. Please contact a doctor now, "
    "or go to the nearest hospital."
)


def render_followup(result_key: str, language: str, has_key: bool) -> None:
    """Chat-style follow-up questions grounded in the stored explanation."""
    result = st.session_state.get(result_key)
    if not result:
        return
    history = result.setdefault("followups", [])
    st.divider()
    st.subheader("💬 Ask a follow-up question")
    for item in history:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            if item.get("see_doctor"):
                st.error(EMERGENCY_MESSAGE, icon="🚑")
            st.markdown(item["a"])

    remaining = MAX_FOLLOWUPS - len(history)
    if remaining <= 0:
        st.caption(f"You've asked {MAX_FOLLOWUPS} questions about this result. Please discuss anything else with your doctor.")
        return
    with st.form(f"{result_key}_followup", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            max_chars=MAX_QUESTION_CHARS,
            placeholder=FOLLOWUP_PLACEHOLDERS.get(result_key, ""),
            key=f"{result_key}_question",
        )
        asked = st.form_submit_button("Ask", disabled=not has_key)
    st.caption(
        f"Answers use this result and general health information. Not medical advice. {remaining} question(s) left."
    )
    if asked and question.strip():
        limit_msg = try_use_quota()
        if limit_msg:
            st.warning(limit_msg)
            return
        try:
            with st.spinner("Thinking…"):
                answer = ask_followup(result["summary"], question, history, language)
            history.append({"q": question.strip(), "a": answer.answer, "see_doctor": answer.see_doctor_soon})
            st.rerun()
        except SummarizerError as e:
            st.error(str(e))


def stale_language_note(result: dict, language: str) -> None:
    if result["language"] != language:
        st.caption(f"Showing results in {result['language']}. Click the button again to get them in {language}.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def clear(key: str):
    return lambda: st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="Lab Report Explainer", page_icon="🧪", layout="centered")

    with st.sidebar:
        language = st.selectbox("Summary language", LANGUAGES, index=0, key="language")
        accurate = st.toggle(
            "Accurate mode (slower)",
            value=False,
            key="accurate",
            help="Gemini thinks longer before answering. Use it for blurry, scanned or unusual reports.",
        )
        mode = "accurate" if accurate else "fast"
        st.divider()
        st.markdown(
            "**How it works**\n\n"
            "1. Upload a lab report, scan report (X-ray, CT, MRI…) or a photo of a prescription\n"
            "2. Gemini reads it, including scanned pages and handwriting\n"
            "3. You get a plain-language explanation, and can ask follow-up questions\n\n"
            "**Privacy:** files are sent to the Gemini API for processing and are not saved by this app."
        )
        st.caption(f"Model: {os.environ.get('GEMINI_MODEL', DEFAULT_MODEL)}")

    st.title("🧪 Lab Report Explainer")
    st.write("Upload a lab report, a scan report or a prescription and get it explained in plain language.")
    st.info(DISCLAIMER, icon="ℹ️")
    if demo_mode():
        st.warning(DEMO_BANNER, icon="🧪")

    has_key = bool(get_api_key())
    if not has_key:
        st.error("No Gemini API key configured. Add `GEMINI_API_KEY=...` to a `.env` file and restart the app.")

    tab_single, tab_compare, tab_scan, tab_note = st.tabs(
        ["📄 Lab report", "📈 Compare reports", "🩻 Scans & other reports", "💊 Prescription"]
    )

    with tab_single:
        pdf = st.file_uploader(
            "Lab report (PDF or photo)", type=UPLOAD_TYPES, key="single_file", on_change=clear("single_result")
        )
        b1, b2 = st.columns(2)
        explain = b1.button("Explain my report", type="primary", disabled=not (pdf and has_key), key="single_btn")
        sample = b2.button(
            "Try a sample report", disabled=not has_key, key="single_sample", help="A made-up report from a fake lab"
        )
        data = pdf.getvalue() if explain else SAMPLE_EARLIER.read_bytes() if sample else None
        if data:
            limit_msg = try_use_quota()
            if limit_msg:
                st.warning(limit_msg)
            else:
                try:
                    wait = "up to a minute" if accurate else "about 20–40 seconds"
                    with st.spinner(f"Reading your report… this can take {wait}."):
                        summary, corrected = summarize_report(data, language, mode)
                    st.session_state["single_result"] = {
                        "summary": summary.model_dump(),
                        "corrected": corrected,
                        "language": language,
                    }
                except SummarizerError as e:
                    st.error(str(e))

        result = st.session_state.get("single_result")
        if result:
            stale_language_note(result, language)
            render_summary(ReportSummary.model_validate(result["summary"]), result.get("corrected"))
            if result["summary"]["is_lab_report"]:
                render_followup("single_result", language, has_key)

    with tab_compare:
        st.write("See how your results changed between two reports.")
        c1, c2 = st.columns(2)
        earlier = c1.file_uploader(
            "Earlier report", type=UPLOAD_TYPES, key="earlier_file", on_change=clear("compare_result")
        )
        later = c2.file_uploader("Later report", type=UPLOAD_TYPES, key="later_file", on_change=clear("compare_result"))
        b1, b2 = st.columns(2)
        compare = b1.button(
            "Compare reports", type="primary", disabled=not (earlier and later and has_key), key="compare_btn"
        )
        sample = b2.button(
            "Try with sample reports", disabled=not has_key, key="compare_sample", help="Two made-up reports, 6 months apart"
        )
        pair = (
            (earlier.getvalue(), later.getvalue())
            if compare
            else (SAMPLE_EARLIER.read_bytes(), SAMPLE_LATER.read_bytes())
            if sample
            else None
        )
        if pair:
            limit_msg = try_use_quota()
            if limit_msg:
                st.warning(limit_msg)
            else:
                try:
                    with st.spinner("Comparing your reports… this can take up to a minute."):
                        cmp = compare_reports(*pair, language, mode)
                    st.session_state["compare_result"] = {"summary": cmp.model_dump(), "language": language}
                except SummarizerError as e:
                    st.error(str(e))

        result = st.session_state.get("compare_result")
        if result:
            stale_language_note(result, language)
            render_comparison(ComparisonSummary.model_validate(result["summary"]))
            if result["summary"]["both_lab_reports"]:
                render_followup("compare_result", language, has_key)

    with tab_scan:
        st.write(
            "Upload the written report from an X-ray, CT, MRI, ultrasound, ECG, echo or biopsy "
            "and get it explained in plain words."
        )
        st.caption(SCAN_WARNING)
        scan_file = st.file_uploader(
            "Scan or test report (PDF or photo)", type=UPLOAD_TYPES, key="scan_file", on_change=clear("scan_result")
        )
        b1, b2 = st.columns(2)
        explain = b1.button("Explain this report", type="primary", disabled=not (scan_file and has_key), key="scan_btn")
        sample = b2.button(
            "Try a sample MRI report", disabled=not has_key, key="scan_sample", help="A made-up knee MRI report"
        )
        data = scan_file.getvalue() if explain else SAMPLE_SCAN.read_bytes() if sample else None
        if data:
            limit_msg = try_use_quota()
            if limit_msg:
                st.warning(limit_msg)
            else:
                try:
                    with st.spinner("Reading the report… this can take 20–60 seconds."):
                        scan = explain_scan(data, language, mode)
                    st.session_state["scan_result"] = {"summary": scan.model_dump(), "language": language}
                except SummarizerError as e:
                    st.error(str(e))

        result = st.session_state.get("scan_result")
        if result:
            stale_language_note(result, language)
            render_scan(ScanSummary.model_validate(result["summary"]))
            if result["summary"]["document_kind"] == "written_report":
                render_followup("scan_result", language, has_key)

    with tab_note:
        st.write("Take a clear photo of a prescription or doctor's note and get it explained in plain words.")
        st.warning(NOTE_WARNING, icon="⚠️")
        st.caption(
            f"Tips: good light, no shadows, the whole page in frame, held flat. Up to {MAX_NOTE_FILES} photos "
            "for a multi-page note. On a phone, tap Browse files to take a photo."
        )
        photos = st.file_uploader(
            "Photos of the note",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key="note_files",
            on_change=clear("note_result"),
        )
        if st.button("Explain this note", type="primary", disabled=not (photos and has_key), key="note_btn"):
            limit_msg = try_use_quota()
            if limit_msg:
                st.warning(limit_msg)
            else:
                try:
                    with st.spinner("Reading the handwriting carefully… this can take up to a minute."):
                        note, codes = explain_note([f.getvalue() for f in photos], language)
                    st.session_state["note_result"] = {
                        "summary": note.model_dump(),
                        "codes": codes,
                        "language": language,
                    }
                except SummarizerError as e:
                    st.error(str(e))

        result = st.session_state.get("note_result")
        if result:
            stale_language_note(result, language)
            render_note(NoteSummary.model_validate(result["summary"]), result["codes"])
            if result["summary"]["is_medical_note"]:
                render_followup("note_result", language, has_key)


main()

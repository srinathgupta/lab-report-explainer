"""Scan and other written test reports tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import summarizer as core

from .common import SAMPLES_DIR, clear, render_followup, run_job, stale_language_note

SAMPLE_SCAN = SAMPLES_DIR / "sample_mri_knee.pdf"

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


def render_scan(scan: core.ScanSummary) -> None:
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


def scan_tab(language: str, mode: str, has_key: bool) -> None:
    st.write(
        "Upload the written report from an X-ray, CT, MRI, ultrasound, ECG, echo or biopsy "
        "and get it explained in plain words."
    )
    st.caption(SCAN_WARNING)
    scan_file = st.file_uploader(
        "Scan or test report (PDF or photo)", type=core.UPLOAD_TYPES, key="scan_file", on_change=clear("scan_result")
    )
    b1, b2 = st.columns(2)
    explain = b1.button("Explain this report", type="primary", disabled=not (scan_file and has_key), key="scan_btn")
    sample = b2.button(
        "Try a sample MRI report", disabled=not has_key, key="scan_sample", help="A made-up knee MRI report"
    )
    data = scan_file.getvalue() if explain else SAMPLE_SCAN.read_bytes() if sample else None
    if data:
        run_job(
            "scan_result",
            "Reading the report… this can take 20–60 seconds.",
            lambda: {"summary": core.explain_scan(data, language, mode).model_dump(), "language": language},
        )

    result = st.session_state.get("scan_result")
    if result:
        stale_language_note(result, language)
        render_scan(core.ScanSummary.model_validate(result["summary"]))
        if result["summary"]["document_kind"] == "written_report":
            render_followup("scan_result", language, has_key)

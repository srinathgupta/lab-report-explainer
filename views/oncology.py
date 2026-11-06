"""Cancer reports tab: biopsy / pathology reports and tumour-marker tests."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import summarizer as core

from .common import SAMPLES_DIR, clear, render_followup, run_job, stale_language_note
from .lab import STATUS_LABELS

SAMPLE_ONCO = SAMPLES_DIR / "sample_breast_biopsy.pdf"

ONCO_INTRO = (
    "Getting a cancer report can feel overwhelming. This page explains the words in your report so your next "
    "conversation with your oncologist is easier. It cannot tell you what will happen. Your care team knows your "
    "full situation."
)
SUPPORT_MESSAGE = (
    "You don't have to go through this alone. Your care team, hospital counsellors and patient support groups "
    "can help you and your family."
)
GLOSSARY_NOTE = (
    "📖 These meanings come from a standard glossary built into the app, not from AI. The stage on its own "
    "doesn't decide what happens next. Ask your oncologist what it means for you."
)
REMOVED_NOTE = (
    "Some statements about outlook or survival were removed on purpose. Your oncologist is the right person "
    "to talk with about what to expect."
)
KIND_LABELS = {
    "pathology_report": "Biopsy / pathology report",
    "tumour_marker_report": "Tumour marker test",
    "other_cancer_report": "Cancer report",
}


def _item(label: str, item) -> None:
    if item and item.as_written:
        st.markdown(f"**{label}:** {item.as_written}  \n{item.plain_explanation}")


def render_staging(staging) -> None:
    stage = core.explain_stage(staging.stage_group_as_written)
    tnm = core.explain_tnm(staging.tnm_as_written)
    grade = core.explain_grade(staging.grade_as_written)
    if not (staging.stage_group_as_written or staging.tnm_as_written or staging.grade_as_written):
        return
    st.subheader("Stage and grade")
    if staging.stage_group_as_written:
        st.markdown(f"**Stage:** {staging.stage_group_as_written}")
        if stage:
            st.markdown(f"{stage[0]} usually means {stage[1]}.")
    if staging.tnm_as_written:
        st.markdown(f"**TNM:** `{staging.tnm_as_written}`")
        if tnm:
            st.markdown("\n".join(f"- `{c}`: {m}" for c, m in tnm))
    if staging.grade_as_written:
        st.markdown(f"**Grade:** {staging.grade_as_written}")
        if grade:
            st.markdown(f"This means {grade[1]}.")
    if stage or tnm or grade:
        st.caption(GLOSSARY_NOTE)


def render_onco(onco, removed: int = 0, corrected: list[str] | None = None) -> None:
    if onco.document_kind == "scan_image":
        st.warning(
            "This looks like a scan picture, not a written report. This app only explains written reports. "
            "Reading scan images needs a trained radiologist.",
            icon="🩻",
        )
        return
    if onco.document_kind == "not_cancer_related":
        st.warning(
            "This doesn't look like a cancer biopsy or tumour-marker report. "
            "For other reports, try the Lab report or Scans tabs."
        )
        return

    st.caption(" · ".join(x for x in [KIND_LABELS.get(onco.document_kind, ""), onco.organ, onco.date] if x))
    if onco.cancer_type_as_written:
        st.markdown(f"**Diagnosis as written:** {onco.cancer_type_as_written}")

    st.subheader("Summary")
    st.write(onco.overall_summary)

    render_staging(onco.staging)

    if any(x and x.as_written for x in (onco.tumour_size, onco.lymph_nodes, onco.margins)):
        st.subheader("Details from the report")
        _item("Tumour size", onco.tumour_size)
        _item("Lymph nodes", onco.lymph_nodes)
        _item("Margins", onco.margins)

    if onco.biomarkers:
        st.subheader("Receptors and markers in the tumour")
        for b in onco.biomarkers:
            st.markdown(f"**{b.name}:** {b.result_as_written}  \n{b.plain_explanation}")
            info = core.glossary_for(b.name, core.BIOMARKER_GLOSSARY)
            if info:
                st.caption(f"📖 {info}")

    if onco.tumour_markers:
        st.subheader("Tumour marker blood tests")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Test": t.name,
                        "Result": f"{t.value} {t.unit}".strip(),
                        "Reference range": t.reference_range or "—",
                        "Status": STATUS_LABELS.get(t.status, t.status),
                    }
                    for t in onco.tumour_markers
                ]
            ),
            hide_index=True,
        )
        for t in onco.tumour_markers:
            st.markdown(f"**{t.name}:** {t.plain_explanation}")
            info = core.glossary_for(t.name, core.TUMOUR_MARKER_GLOSSARY)
            if info:
                st.caption(f"📖 {info}")
        st.info(core.TUMOUR_MARKER_NOTE, icon="📈")
        if corrected:
            st.caption(f"Status checked against the printed range and corrected for: {', '.join(corrected)}.")

    if onco.key_points:
        st.subheader("Other key points")
        for k in onco.key_points:
            st.markdown(f"**{k.as_written}**  \n{k.plain_explanation}")

    if onco.terms:
        st.subheader("Medical words explained")
        st.dataframe(
            pd.DataFrame([{"Word": x.as_written, "Meaning": x.plain_explanation} for x in onco.terms]), hide_index=True
        )

    if onco.questions_for_doctor:
        st.subheader("Questions to ask your oncologist")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(onco.questions_for_doctor, 1)))
    if onco.notes:
        st.info(onco.notes, icon="📝")
    if removed:
        st.caption(REMOVED_NOTE)
    st.info(SUPPORT_MESSAGE, icon="🎗️")


def onco_tab(language: str, mode: str, has_key: bool) -> None:
    st.write("Upload a biopsy / pathology report or tumour-marker test and get it explained in plain words.")
    st.info(ONCO_INTRO, icon="🎗️")
    onco_file = st.file_uploader(
        "Cancer report (PDF or photo)", type=core.UPLOAD_TYPES, key="onco_file", on_change=clear("onco_result")
    )
    b1, b2 = st.columns(2)
    explain = b1.button("Explain this report", type="primary", disabled=not (onco_file and has_key), key="onco_btn")
    sample = b2.button(
        "Try a sample biopsy report", disabled=not has_key, key="onco_sample", help="A made-up breast biopsy report"
    )
    data = onco_file.getvalue() if explain else SAMPLE_ONCO.read_bytes() if sample else None
    if data:
        def job():
            # Cancer reports always use careful mode: these details matter.
            onco, removed, corrected = core.explain_oncology(data, language, "accurate")
            return {"summary": onco.model_dump(), "removed": removed, "corrected": corrected, "language": language}

        run_job("onco_result", "Reading the report carefully… this can take up to a minute.", job)

    result = st.session_state.get("onco_result")
    if result:
        stale_language_note(result, language)
        render_onco(core.OncologySummary.model_validate(result["summary"]), result.get("removed", 0), result.get("corrected"))
        if result["summary"]["document_kind"] not in ("scan_image", "not_cancer_related"):
            render_followup("onco_result", language, has_key)

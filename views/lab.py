"""Lab report and report-comparison tabs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import summarizer as core

from .common import SAMPLES_DIR, clear, render_followup, run_job, stale_language_note

SAMPLE_EARLIER = SAMPLES_DIR / "sample_report_2026-03.pdf"
SAMPLE_LATER = SAMPLES_DIR / "sample_report_2026-09.pdf"

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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_summary(summary: core.ReportSummary, corrected: list[str] | None = None) -> None:
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


def render_comparison(cmp: core.ComparisonSummary) -> None:
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


def lab_tab(language: str, mode: str, has_key: bool) -> None:
    pdf = st.file_uploader(
        "Lab report (PDF or photo)", type=core.UPLOAD_TYPES, key="single_file", on_change=clear("single_result")
    )
    b1, b2 = st.columns(2)
    explain = b1.button("Explain my report", type="primary", disabled=not (pdf and has_key), key="single_btn")
    sample = b2.button(
        "Try a sample report", disabled=not has_key, key="single_sample", help="A made-up report from a fake lab"
    )
    data = pdf.getvalue() if explain else SAMPLE_EARLIER.read_bytes() if sample else None
    if data:
        def job():
            summary, corrected = core.summarize_report(data, language, mode)
            return {"summary": summary.model_dump(), "corrected": corrected, "language": language}

        wait = "up to a minute" if mode == "accurate" else "about 20–40 seconds"
        run_job("single_result", f"Reading your report… this can take {wait}.", job)

    result = st.session_state.get("single_result")
    if result:
        stale_language_note(result, language)
        render_summary(core.ReportSummary.model_validate(result["summary"]), result.get("corrected"))
        if result["summary"]["is_lab_report"]:
            render_followup("single_result", language, has_key)


def compare_tab(language: str, mode: str, has_key: bool) -> None:
    st.write("See how your results changed between two reports.")
    c1, c2 = st.columns(2)
    earlier = c1.file_uploader(
        "Earlier report", type=core.UPLOAD_TYPES, key="earlier_file", on_change=clear("compare_result")
    )
    later = c2.file_uploader("Later report", type=core.UPLOAD_TYPES, key="later_file", on_change=clear("compare_result"))
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
        run_job(
            "compare_result",
            "Comparing your reports… this can take up to a minute.",
            lambda: {"summary": core.compare_reports(*pair, language, mode).model_dump(), "language": language},
        )

    result = st.session_state.get("compare_result")
    if result:
        stale_language_note(result, language)
        render_comparison(core.ComparisonSummary.model_validate(result["summary"]))
        if result["summary"]["both_lab_reports"]:
            render_followup("compare_result", language, has_key)

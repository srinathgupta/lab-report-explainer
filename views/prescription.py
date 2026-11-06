"""Prescription / doctor's note tab."""

from __future__ import annotations

import streamlit as st

import summarizer as core

from .common import clear, render_followup, run_job, stale_language_note

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


def render_note(note: core.NoteSummary, codes: dict) -> None:
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


def note_tab(language: str, mode: str, has_key: bool) -> None:
    st.write("Take a clear photo of a prescription or doctor's note and get it explained in plain words.")
    st.warning(NOTE_WARNING, icon="⚠️")
    st.caption(
        f"Tips: good light, no shadows, the whole page in frame, held flat. Up to {core.MAX_NOTE_FILES} photos "
        "for a multi-page note. On a phone, tap Browse files to take a photo."
    )
    photos = st.file_uploader(
        "Photos of the note",
        type=core.UPLOAD_TYPES,
        accept_multiple_files=True,
        key="note_files",
        on_change=clear("note_result"),
    )
    if st.button("Explain this note", type="primary", disabled=not (photos and has_key), key="note_btn"):
        def job():
            note, codes = core.explain_note([f.getvalue() for f in photos], language)
            return {"summary": note.model_dump(), "codes": codes, "language": language}

        run_job("note_result", "Reading the handwriting carefully… this can take up to a minute.", job)

    result = st.session_state.get("note_result")
    if result:
        stale_language_note(result, language)
        render_note(core.NoteSummary.model_validate(result["summary"]), result["codes"])
        if result["summary"]["is_medical_note"]:
            render_followup("note_result", language, has_key)

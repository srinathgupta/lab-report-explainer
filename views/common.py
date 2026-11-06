"""Shared UI helpers: usage limits, running a Gemini job, follow-up chat."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import streamlit as st

import summarizer as core

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "sample_reports"


def clear(key: str):
    return lambda: st.session_state.pop(key, None)


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


def run_job(result_key: str, spinner: str, job: Callable[[], dict]) -> None:
    """Check usage limits, run a Gemini job, store its result, and show friendly errors."""
    limit_msg = try_use_quota()
    if limit_msg:
        st.warning(limit_msg)
        return
    try:
        with st.spinner(spinner):
            st.session_state[result_key] = job()
    except core.SummarizerError as e:
        st.error(str(e))


FOLLOWUP_PLACEHOLDERS = {
    "single_result": "e.g. What can help bring my LDL cholesterol down?",
    "compare_result": "e.g. Why might my vitamin D have improved?",
    "scan_result": "e.g. What does 'joint effusion' mean?",
    "note_result": "e.g. What is a CBC test?",
    "onco_result": "e.g. What does HER2 negative mean?",
}
EMERGENCY_MESSAGE = (
    "What you describe may need prompt medical attention. Please contact a doctor now, "
    "or go to the nearest hospital."
)


def _followup_options(result_key: str) -> dict:
    # Cancer follow-ups never discuss survival or outlook; the code removes such sentences too.
    return {"no_prognosis": True} if result_key == "onco_result" else {}


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

    remaining = core.MAX_FOLLOWUPS - len(history)
    if remaining <= 0:
        st.caption(f"You've asked {core.MAX_FOLLOWUPS} questions about this result. Please discuss anything else with your doctor.")
        return
    with st.form(f"{result_key}_followup", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            max_chars=core.MAX_QUESTION_CHARS,
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
                answer = core.ask_followup(
                    result["summary"], question, history, language, **_followup_options(result_key)
                )
            history.append({"q": question.strip(), "a": answer.answer, "see_doctor": answer.see_doctor_soon})
            st.rerun()
        except core.SummarizerError as e:
            st.error(str(e))


def stale_language_note(result: dict, language: str) -> None:
    if result["language"] != language:
        st.caption(f"Showing results in {result['language']}. Click the button again to get them in {language}.")

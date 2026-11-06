"""Lab Report Explainer — Streamlit entry point.

Run with:  streamlit run app.py
Each tab lives in views/; the logic it calls lives in medexplain/ (via summarizer.py).
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

import summarizer as core
from views.common import try_use_quota  # noqa: F401  (re-exported for tests)
from views.lab import compare_tab, lab_tab
from views.oncology import onco_tab
from views.prescription import note_tab
from views.scan import scan_tab
from views.tracker import tracker_tab

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


def main() -> None:
    st.set_page_config(page_title="Lab Report Explainer", page_icon="🧪", layout="centered")

    with st.sidebar:
        language = st.selectbox("Summary language", core.LANGUAGES, index=0, key="language")
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
            "1. Upload a lab report, scan report (X-ray, CT, MRI…), cancer report, or a photo of a prescription\n"
            "2. Gemini reads it, including scanned pages and handwriting\n"
            "3. You get a plain-language explanation, and can ask follow-up questions\n\n"
            "**Privacy:** files are sent to the Gemini API for processing and are not saved by this app."
        )
        st.caption(f"Model: {os.environ.get('GEMINI_MODEL', core.DEFAULT_MODEL)}")

    st.title("🧪 Lab Report Explainer")
    st.write("Upload a lab report, scan report, cancer report or prescription and get it explained in plain language.")
    st.info(DISCLAIMER, icon="ℹ️")
    if demo_mode():
        st.warning(DEMO_BANNER, icon="🧪")

    has_key = bool(core.get_api_key())
    if not has_key:
        st.error("No Gemini API key configured. Add `GEMINI_API_KEY=...` to a `.env` file and restart the app.")

    tab_single, tab_compare, tab_scan, tab_onco, tab_track, tab_note = st.tabs(
        ["📄 Lab report", "📈 Compare", "🩻 Scans", "🎗️ Cancer reports", "📝 Side effects", "💊 Prescription"]
    )

    with tab_single:
        lab_tab(language, mode, has_key)
    with tab_compare:
        compare_tab(language, mode, has_key)
    with tab_scan:
        scan_tab(language, mode, has_key)
    with tab_onco:
        onco_tab(language, mode, has_key)
    with tab_track:
        tracker_tab(language, mode, has_key)
    with tab_note:
        note_tab(language, mode, has_key)


main()

"""Side-effect tracker tab. Plain code only: no AI, no API key, nothing stored on a server."""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import summarizer as core

RULES_NOTE = (
    "The warning signs below are general guidance used by many cancer centres. **Your own care team's "
    "instructions and emergency number always come first.** If you feel very unwell, don't wait for this app: "
    "contact them straight away."
)
PRIVACY_NOTE = (
    "Your log is **not stored by this app**. After each update, tap **Save my log** to download it, "
    "then load that file next time to continue."
)
ALERT_STYLES = {
    "emergency": (st.error, "🚑"),
    "urgent": (st.error, "⚠️"),
    "soon": (st.warning, "📞"),
    "watch": (st.info, "🌡️"),
}


def _log() -> core.Log:
    return st.session_state.setdefault("tracker_log", core.Log())


def render_alerts(alerts: list, day: date) -> None:
    st.subheader(f"Warning signs · {day.strftime('%d %b %Y')}")
    if not alerts:
        st.success("No warning signs in your latest entry. Keep following your care team's advice.", icon="✅")
        return
    for a in alerts:
        show, icon = ALERT_STYLES[a.level]
        show(a.message, icon=icon)


def temperature_chart(temps: list[tuple[date, float]]):
    df = pd.DataFrame(temps, columns=["Date", "Temperature"])
    df["Date"] = pd.to_datetime(df["Date"])
    lo = min(35.0, df["Temperature"].min() - 0.5)
    hi = max(40.0, df["Temperature"].max() + 0.5)
    scale = alt.Scale(domain=[lo, hi])
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b", tickCount={"interval": "day", "step": 1})),
        y=alt.Y("Temperature:Q", title="Temperature (°C)", scale=scale),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Temperature:Q", format=".1f")],
    )
    fever = alt.Chart(pd.DataFrame({"y": [core.FEVER_C]})).mark_rule(color="#d62728", strokeDash=[6, 4]).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y": [core.FEVER_C], "t": [f"Fever {core.FEVER_C:.0f}°C"]})).mark_text(
        align="left", dx=4, dy=-7, color="#d62728").encode(y="y:Q", text="t:N", x=alt.value(0))
    return (line + fever + label).properties(height=220, title="Temperature")


def render_trends(log: core.Log) -> None:
    entries = log.sorted_entries()
    if len(entries) < 2:
        st.caption("Log at least two days to see your trends.")
        return
    st.subheader("Your trends")
    temps = [(e.day, e.temperature_c) for e in entries if e.temperature_c is not None]
    if temps:
        st.altair_chart(temperature_chart(temps), use_container_width=True)
    scores = pd.DataFrame(
        [{"Date": e.day, **{label: e.scores.get(k, 0) for k, (label, _) in core.SCALE_SYMPTOMS.items()}} for e in entries]
    ).set_index("Date")
    scores = scores.loc[:, (scores > 0).any()]
    if not scores.empty:
        st.caption("Symptoms: 0 none · 1 mild · 2 moderate · 3 severe · 4 very severe")
        st.line_chart(scores, height=260)


def render_history(log: core.Log) -> None:
    entries = log.sorted_entries()
    with st.expander(f"All entries ({len(entries)})"):
        rows = []
        for e in reversed(entries):
            worst = max(e.scores.items(), key=lambda kv: kv[1], default=(None, 0))
            rows.append({
                "Date": e.day.strftime("%d %b %Y"),
                "Temp °C": f"{e.temperature_c:.1f}" if e.temperature_c is not None else "—",
                "Worst symptom": f"{core.SCALE_SYMPTOMS[worst[0]][0]} ({worst[1]})" if worst[1] else "—",
                "Warnings": len([a for a in core.check_entry(e) if a.level in ("emergency", "urgent")]),
                "Notes": e.notes,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        days = [e.day for e in reversed(entries)]
        c1, c2 = st.columns([3, 1])
        to_remove = c1.selectbox("Remove an entry", days, format_func=lambda d: d.strftime("%d %b %Y"), key="tr_remove_day")
        if c2.button("Remove", key="tr_remove_btn"):
            st.session_state["tracker_log"] = core.remove_entry(log, to_remove)
            st.rerun()


def entry_form() -> core.Entry | None:
    st.subheader("How are you today?")
    with st.form("tracker_form"):
        day = st.date_input("Date", value=date.today(), max_value=date.today(), key="tr_day")
        c1, c2 = st.columns(2)
        took_temp = c1.checkbox("I measured my temperature", key="tr_has_temp")
        temp = c2.number_input("Temperature (°C)", min_value=34.0, max_value=43.0, value=37.0, step=0.1,
                               format="%.1f", key="tr_temp", help="100.4°F = 38.0°C")

        st.markdown("**Symptoms** (0 none · 1 mild · 2 moderate · 3 severe · 4 very severe)")
        cols = st.columns(2)
        scores = {}
        for i, (key, (label, hint)) in enumerate(core.SCALE_SYMPTOMS.items()):
            scores[key] = cols[i % 2].select_slider(
                label, options=[0, 1, 2, 3, 4], value=0, key=f"tr_{key}",
                format_func=lambda v: f"{v} · {core.SCALE_LABELS[v]}", help=hint or None,
            )

        st.markdown("**In the last 24 hours**")
        cols = st.columns(2)
        counts = {
            key: cols[i % 2].number_input(label, min_value=0, max_value=50, value=0, step=1, key=f"tr_{key}")
            for i, (key, label) in enumerate(core.COUNT_SYMPTOMS.items())
        }
        flags = {key: st.checkbox(label, key=f"tr_{key}") for key, label in core.FLAG_SYMPTOMS.items()}
        notes = st.text_area("Notes (optional)", max_chars=core.MAX_NOTE_CHARS, key="tr_notes",
                             placeholder="e.g. Chemo cycle 2, day 3. Ate only soup.")
        if not st.form_submit_button("Save this day", type="primary"):
            return None
    return core.Entry(
        day=day, temperature_c=float(temp) if took_temp else None, scores=scores,
        counts={k: int(v) for k, v in counts.items()}, flags=flags, notes=notes.strip(),
    )


def tracker_tab(language: str, mode: str, has_key: bool) -> None:
    st.write(
        "Keep a simple daily record of how you feel during cancer treatment. The app points out warning signs "
        "and makes a summary to show your oncologist at your next visit."
    )
    st.info(RULES_NOTE, icon="🩺")
    log = _log()

    with st.expander("📂 Continue a saved log", expanded=not log.entries):
        upload = st.file_uploader("Your log file (.json)", type=["json"], key="tracker_upload")
        if upload and st.session_state.get("tracker_loaded") != (upload.name, upload.size):
            try:
                log = core.load_log(upload.getvalue())
                st.session_state["tracker_log"] = log
                st.session_state["tracker_loaded"] = (upload.name, upload.size)
                st.success(f"Loaded {len(log.entries)} day(s).")
            except core.SummarizerError as e:
                st.error(str(e))

    saved = st.session_state.pop("tracker_saved", None)
    if saved:
        st.success(f"Saved {saved.strftime('%d %b %Y')}. Remember to tap **Save my log** below.", icon="💾")
    if log.entries:
        latest = log.sorted_entries()[-1]
        render_alerts(core.latest_alerts(log), latest.day)

    entry = entry_form()
    if entry:
        try:
            st.session_state["tracker_log"] = core.add_entry(log, entry)
            st.session_state["tracker_saved"] = entry.day
            st.rerun()
        except core.SummarizerError as e:
            st.error(str(e))

    if not log.entries:
        st.caption(PRIVACY_NOTE)
        return

    render_trends(log)
    render_history(log)

    st.subheader("Save and share")
    c1, c2 = st.columns(2)
    stamp = latest.day.isoformat()
    c1.download_button("⬇️ Save my log (.json)", core.dump_log(log), file_name=f"side-effect-log-{stamp}.json",
                       mime="application/json", key="tr_save")
    c2.download_button("📄 Summary for my doctor (PDF)", core.doctor_summary_pdf(log),
                       file_name=f"side-effect-summary-{stamp}.pdf", mime="application/pdf", key="tr_pdf")
    st.caption(PRIVACY_NOTE)

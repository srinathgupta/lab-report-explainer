import io
from datetime import date, timedelta
from pathlib import Path

import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import summarizer as s

APP = str(Path(__file__).resolve().parent.parent / "app.py")
D0 = date(2026, 11, 1)


def entry(day=0, **kw):
    return s.Entry(day=D0 + timedelta(days=day), **kw)


def rules(e):
    return [(a.level, a.rule) for a in s.check_entry(e)]


# --- warning rules ----------------------------------------------------------


@pytest.mark.parametrize("kw,expected", [
    ({}, []),
    ({"temperature_c": 37.4}, []),
    ({"temperature_c": 37.5}, [("watch", "warm")]),
    ({"temperature_c": 37.9}, [("watch", "warm")]),
    ({"temperature_c": 38.0}, [("urgent", "fever")]),
    ({"temperature_c": 39.2}, [("urgent", "fever")]),
    ({"flags": {"chest_pain": True}}, [("emergency", "breathing")]),
    ({"scores": {"breathlessness": 3}}, [("emergency", "breathing")]),
    ({"scores": {"breathlessness": 2}}, []),
    ({"flags": {"bleeding": True}}, [("urgent", "bleeding")]),
    ({"flags": {"confusion": True}}, [("urgent", "confusion")]),
    ({"flags": {"cannot_drink": True}}, [("urgent", "vomiting")]),
    ({"counts": {"vomiting_times": 2}}, []),
    ({"counts": {"vomiting_times": 3}}, [("urgent", "vomiting")]),
    ({"counts": {"loose_stools": 3}}, []),
    ({"counts": {"loose_stools": 4}}, [("urgent", "diarrhoea")]),
    ({"scores": {"nausea": 2}}, []),
    ({"scores": {"nausea": 3}}, [("soon", "nausea_3")]),
    ({"scores": {"pain": 4}}, [("urgent", "pain_4")]),
])
def test_single_day_rules(kw, expected):
    assert rules(entry(**kw)) == expected


def test_alerts_sorted_most_serious_first():
    e = entry(temperature_c=38.5, scores={"nausea": 3}, flags={"chest_pain": True})
    assert [a.level for a in s.check_entry(e)] == ["emergency", "urgent", "soon"]


def test_fever_message_and_emergency_number():
    (fever,) = s.check_entry(entry(temperature_c=38.4))
    assert "38.4°C" in fever.message and "any time of day or night" in fever.message
    assert fever.title == "Temperature 38.4°C"
    (emerg,) = s.check_entry(entry(flags={"chest_pain": True}))
    assert "112" in emerg.message and "108" in emerg.message


def test_trend_needs_consecutive_days():
    log = s.Log()
    for d, score in [(0, 2), (1, 3), (2, 2)]:
        log = s.add_entry(log, entry(d, scores={"tiredness": score}))
    assert [a.rule for a in s.check_trends(log)] == ["tiredness_trend"]
    # a gap of a day breaks the run
    gap = s.Log()
    for d in (0, 1, 3):
        gap = s.add_entry(gap, entry(d, scores={"tiredness": 2}))
    assert s.check_trends(gap) == []
    # trend must still be going on the latest day
    ended = s.add_entry(log, entry(3, scores={"tiredness": 0}))
    assert s.check_trends(ended) == []


# --- log handling -----------------------------------------------------------


def test_add_entry_replaces_same_day_and_sorts():
    log = s.add_entry(s.Log(), entry(2, notes="b"))
    log = s.add_entry(log, entry(0, notes="a"))
    log = s.add_entry(log, entry(2, notes="c"))
    assert [(e.day.day, e.notes) for e in log.entries] == [(1, "a"), (3, "c")]
    assert [e.day.day for e in s.remove_entry(log, D0).entries] == [3]


def test_round_trip_and_bad_files():
    log = s.add_entry(s.Log(), entry(0, temperature_c=38.1, scores={"pain": 2}, counts={"vomiting_times": 1},
                                     flags={"bleeding": True}, notes="cycle 2"))
    assert s.load_log(s.dump_log(log)) == log
    for bad in (b"not json", b'{"app": "something-else"}', b"{" + b" " * (s.MAX_LOG_BYTES + 1) + b"}"):
        with pytest.raises(s.SummarizerError):
            s.load_log(bad)


def test_values_are_clamped_and_unknown_keys_dropped():
    e = s.Entry(day=D0, scores={"pain": 9, "made_up": 3}, counts={"vomiting_times": -2}, flags={"x": True})
    assert e.scores == {"pain": 4} and e.counts == {"vomiting_times": 0} and e.flags == {}
    with pytest.raises(Exception):
        s.Entry(day=D0, temperature_c=60)
    with pytest.raises(Exception):
        s.Entry(day=D0, notes="x" * (s.MAX_NOTE_CHARS + 1))


# --- doctor summary ---------------------------------------------------------


def sample_log():
    log = s.Log()
    for d, (temp, nausea, vomit) in enumerate([(37.0, 1, 0), (38.4, 2, 1), (37.2, 3, 3)]):
        log = s.add_entry(log, entry(d, temperature_c=temp, scores={"nausea": nausea},
                                     counts={"vomiting_times": vomit}, notes="ate soup" if d == 1 else ""))
    return log


def test_summary_numbers():
    r = s.summary_rows(sample_log())
    assert r["days_logged"] == 3 and r["max_temp"] == (D0 + timedelta(days=1), 38.4)
    assert r["fever_days"] == [D0 + timedelta(days=1)] and r["max_vomiting"] == 3
    (nausea,) = r["symptoms"]
    assert (nausea["worst"], nausea["days_moderate"], nausea["latest"]) == (3, 2, 3)
    assert [a.rule for _, a in r["events"]] == ["fever", "vomiting"]


def test_doctor_pdf_contents():
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(s.doctor_summary_pdf(sample_log()))).pages)
    assert "Side-effect summary for my care team" in text
    assert "38.4" in text and "Nausea" in text and "Vomited 3 times" in text and "ate soup" in text
    with pytest.raises(s.SummarizerError, match="empty"):
        s.doctor_summary_pdf(s.Log())


# --- screen -----------------------------------------------------------------


def run_app(monkeypatch, **state):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    at = AppTest.from_file(APP, default_timeout=30)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def save_button(at):
    return next(b for b in at.button if b.label == "Save this day")


def test_save_entry_shows_fever_alert(monkeypatch):
    at = run_app(monkeypatch)
    at.checkbox(key="tr_has_temp").check()
    at.number_input(key="tr_temp").set_value(38.6)
    at.select_slider(key="tr_nausea").set_value(3)
    at.text_area(key="tr_notes").input("cycle 1 day 4")
    save_button(at).click().run()
    assert not at.exception
    log = at.session_state["tracker_log"]
    assert len(log.entries) == 1 and log.entries[0].temperature_c == 38.6 and log.entries[0].notes == "cycle 1 day 4"
    errors = [e.value for e in at.error]
    assert any("Temperature 38.6°C" in e for e in errors)
    assert any("Nausea (feeling sick) is severe" in w.value for w in at.warning)


def test_temperature_ignored_unless_ticked(monkeypatch):
    at = run_app(monkeypatch)
    at.number_input(key="tr_temp").set_value(39.0)
    save_button(at).click().run()
    assert at.session_state["tracker_log"].entries[0].temperature_c is None
    assert any("No warning signs" in x.value for x in at.success)


def test_existing_log_shows_history_and_downloads(monkeypatch):
    at = run_app(monkeypatch, tracker_log=sample_log())
    assert not at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "⬇️ Save my log (.json)" in labels and "📄 Summary for my doctor (PDF)" in labels
    assert any("All entries (3)" in e.label for e in at.expander)
    assert any("Vomiting 3 or more times" in e.value for e in at.error)


def test_remove_entry(monkeypatch):
    at = run_app(monkeypatch, tracker_log=sample_log())
    at.selectbox(key="tr_remove_day").set_value(D0 + timedelta(days=2))
    at.button(key="tr_remove_btn").click().run()
    assert [e.day.day for e in at.session_state["tracker_log"].entries] == [1, 2]

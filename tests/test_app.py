from pathlib import Path

from streamlit.testing.v1 import AppTest

from test_summarizer import COMPARE_JSON, SUMMARY_JSON

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def all_text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.error] + [e.value for e in at.warning]
    parts += [e.value for e in at.caption] + [e.value for e in at.subheader]
    return "\n".join(str(p) for p in parts)


def run(monkeypatch, key="test-key", **state):
    if key:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(APP, default_timeout=30)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_loads_without_key_and_warns(monkeypatch):
    at = run(monkeypatch, key=None)
    assert not at.exception
    assert any("No Gemini API key" in e.value for e in at.error)
    assert all(b.disabled for b in at.button)  # buttons disabled without key/files


def test_renders_single_summary(monkeypatch):
    data = dict(SUMMARY_JSON, urgent_flags=["Potassium is very high."])
    at = run(monkeypatch, single_result={"summary": data, "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "Potassium is very high." in text
    assert "LDL cholesterol is high." in text
    assert "Should I recheck my cholesterol?" in text
    assert [m.value for m in at.metric] == ["2", "1", "1"]
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 2


def test_out_of_range_toggle(monkeypatch):
    at = run(monkeypatch, single_result={"summary": SUMMARY_JSON, "language": "English"})
    at.toggle(key="only_flagged").set_value(True).run()
    df = at.dataframe[0].value
    assert list(df["Test"]) == ["LDL Cholesterol"]


def test_stale_language_note(monkeypatch):
    at = run(monkeypatch, single_result={"summary": SUMMARY_JSON, "language": "English"})
    at.selectbox(key="language").set_value("Hindi").run()
    assert "Click the button again to get them in Hindi" in all_text(at)


def test_not_a_lab_report(monkeypatch):
    data = dict(SUMMARY_JSON, is_lab_report=False)
    at = run(monkeypatch, single_result={"summary": data, "language": "English"})
    assert any("doesn't look like a medical lab report" in w.value for w in at.warning)


def test_renders_comparison(monkeypatch):
    data = dict(COMPARE_JSON, same_person_likely=False)
    at = run(monkeypatch, compare_result={"summary": data, "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "Vitamin D is higher." in text and "LDL rose from 151 to 162." in text
    assert "different people" in text
    df = at.dataframe[0].value
    assert df.iloc[0]["Change"] == "⚠️ Worsened"


def test_reads_key_from_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.secrets["GEMINI_API_KEY"] = "from-secrets"
    at.run()
    assert not at.exception
    assert not any("No Gemini API key" in e.value for e in at.error)


def test_corrected_status_note(monkeypatch):
    at = run(monkeypatch, single_result={"summary": SUMMARY_JSON, "corrected": ["TSH"], "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "corrected the status of: TSH" in text
    assert "Status set by the app's own check" in text


def test_accurate_mode_toggle_exists(monkeypatch):
    at = run(monkeypatch)
    assert at.toggle(key="accurate").value is False


def quota_script():
    import streamlit as st

    import app

    st.session_state.setdefault("msgs", [])
    st.session_state["msgs"].append(app.try_use_quota() or "ok")
    st.write(" | ".join(st.session_state["msgs"]))


def quota_runs(n):
    at = AppTest.from_function(quota_script, default_timeout=30)
    for _ in range(n):
        at.run()
    assert not at.exception
    return at.markdown[-1].value


def test_session_usage_limit(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("SESSION_LIMIT_PER_HOUR", "2")
    monkeypatch.setenv("GLOBAL_LIMIT_PER_HOUR", "100")
    out = quota_runs(3)
    assert out.startswith("ok | ok | You've reached the limit of 2 reports per hour")


def test_global_usage_limit(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("SESSION_LIMIT_PER_HOUR", "100")
    monkeypatch.setenv("GLOBAL_LIMIT_PER_HOUR", "2")
    assert quota_runs(2) == "ok | ok"
    assert quota_runs(1) == "The app is getting a lot of requests right now. Please try again later."


def test_renders_note_with_safety_messages(monkeypatch):
    from test_summarizer import NOTE_JSON

    import summarizer as s

    note = s.NoteSummary.model_validate(NOTE_JSON)
    codes = s.apply_note_safety(note)
    at = run(monkeypatch, note_result={"summary": note.model_dump(), "codes": codes, "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "Parts of this note were hard to read" in text
    assert "Likely medicine:** Paracetamol" in text
    assert "Amoxicillin" not in text  # low-confidence guess is never shown
    assert "`1-0-1` means: 1 in the morning, none in the afternoon, 1 at night" in text
    assert "Couldn't read clearly: Middle letters of the name" in text
    assert "What is the second medicine?" in text
    assert sum("Never start, stop or change a medicine" in str(i.value) for i in at.warning) == 1
    assert any("Never start, stop or change a medicine" in str(i.value) for i in at.info)


def test_not_a_note(monkeypatch):
    from test_summarizer import NOTE_JSON

    data = dict(NOTE_JSON, is_medical_note=False)
    at = run(monkeypatch, note_result={"summary": data, "codes": {}, "language": "English"})
    assert any("doesn't look like a prescription" in w.value for w in at.warning)


def test_demo_banner_only_in_demo_mode(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert not any("Demo app" in w.value for w in run(monkeypatch).warning)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert any("Demo app" in w.value for w in run(monkeypatch).warning)


def test_sample_buttons_send_sample_reports(monkeypatch):
    import streamlit as st

    import summarizer
    from test_summarizer import PDF, PDF2

    st.cache_resource.clear()
    seen = {}

    def fake_summarize(data, language, mode):
        seen["single"] = data
        return summarizer.ReportSummary.model_validate(SUMMARY_JSON), []

    def fake_compare(a, b, language, mode):
        seen["compare"] = (a, b)
        return summarizer.ComparisonSummary.model_validate(COMPARE_JSON)

    monkeypatch.setattr(summarizer, "summarize_report", fake_summarize)
    monkeypatch.setattr(summarizer, "compare_reports", fake_compare)
    at = run(monkeypatch)
    at.button(key="single_sample").click().run()
    at.button(key="compare_sample").click().run()
    assert not at.exception
    assert seen == {"single": PDF, "compare": (PDF, PDF2)}
    assert "LDL cholesterol is high." in all_text(at)
    assert "Vitamin D is higher." in all_text(at)

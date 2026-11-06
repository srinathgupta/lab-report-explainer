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


def test_renders_scan_report(monkeypatch):
    from test_summarizer import SCAN_JSON

    at = run(monkeypatch, scan_result={"summary": SCAN_JSON, "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "MRI · Left knee · 20-Oct-2026" in text
    assert "🟠 Needs attention · **Grade II degeneration" in text
    assert "Do I need physiotherapy?" in text
    assert [m.value for m in at.metric] == ["1", "1"]
    assert any("Extra fluid inside a joint." in str(df.value.values) for df in at.dataframe)
    assert at.text_input(key="scan_result_question")  # follow-up box shown


def test_scan_image_gets_refusal_and_no_followup(monkeypatch):
    from test_summarizer import SCAN_JSON

    import summarizer as s

    scan = s.apply_scan_safety(s.ScanSummary.model_validate(dict(SCAN_JSON, document_kind="scan_image")))
    at = run(monkeypatch, scan_result={"summary": scan.model_dump(), "language": "English"})
    assert any("scan image itself" in w.value for w in at.warning)
    assert "Grade II" not in all_text(at)
    assert not any(t.key == "scan_result_question" for t in at.text_input)


def test_scan_sample_button(monkeypatch):
    import streamlit as st

    import summarizer
    from test_summarizer import SCAN_JSON, SCAN_PDF

    st.cache_resource.clear()
    seen = {}

    def fake_scan(data, language, mode):
        seen["data"] = data
        return summarizer.ScanSummary.model_validate(SCAN_JSON)

    monkeypatch.setattr(summarizer, "explain_scan", fake_scan)
    at = run(monkeypatch)
    at.button(key="scan_sample").click().run()
    assert not at.exception
    assert seen["data"] == SCAN_PDF
    assert "Grade II degeneration" in all_text(at)


def ask_button(at, prefix="single"):
    return next(b for b in at.button if b.label == "Ask" and b.form_id.startswith(prefix))


def test_followup_flow(monkeypatch):
    import streamlit as st

    import summarizer

    st.cache_resource.clear()
    calls = []

    def fake_followup(context, question, history, language):
        calls.append((question, len(history), language))
        urgent = "chest pain" in question
        return summarizer.FollowUpAnswer(answer=f"Answer to: {question}", off_topic=False, see_doctor_soon=urgent)

    monkeypatch.setattr(summarizer, "ask_followup", fake_followup)
    at = run(monkeypatch, single_result={"summary": SUMMARY_JSON, "language": "English"})
    at.text_input(key="single_result_question").input("What is LDL?")
    ask_button(at).click().run()
    at.text_input(key="single_result_question").input("I have chest pain")
    ask_button(at).click().run()
    assert not at.exception
    assert calls == [("What is LDL?", 0, "English"), ("I have chest pain", 1, "English")]
    assert len(at.chat_message) == 4
    assert "Answer to: What is LDL?" in all_text(at)
    assert sum("prompt medical attention" in e.value for e in at.error) == 1
    assert "3 question(s) left" in all_text(at)


def test_followup_limit(monkeypatch):
    history = [{"q": f"q{i}", "a": f"a{i}", "see_doctor": False} for i in range(5)]
    at = run(monkeypatch, single_result={"summary": SUMMARY_JSON, "language": "English", "followups": history})
    assert not at.exception
    assert "You've asked 5 questions" in all_text(at)
    assert not any(t.key == "single_result_question" for t in at.text_input)


def test_no_followup_for_non_lab_report(monkeypatch):
    data = dict(SUMMARY_JSON, is_lab_report=False)
    at = run(monkeypatch, single_result={"summary": data, "language": "English"})
    assert not any(t.key == "single_result_question" for t in at.text_input)


def test_renders_oncology_report(monkeypatch):
    from test_summarizer import ONCO_JSON

    import summarizer as s

    onco, removed, corrected = s.apply_oncology_safety(s.OncologySummary.model_validate(ONCO_JSON))
    at = run(monkeypatch, onco_result={"summary": onco.model_dump(), "removed": removed, "corrected": corrected,
                                       "language": "English"})
    assert not at.exception
    text = all_text(at)
    assert "Diagnosis as written:** Invasive ductal carcinoma (NST)" in text
    assert "This means intermediate grade" in text
    assert "- `pN1a`: cancer was found in nearby lymph nodes" in text  # from the fixed glossary
    assert "📖 Estrogen receptor: whether the cancer cells use the hormone estrogen" in text
    assert "removed on purpose" in text
    assert "corrected for: CA 15-3" in text
    assert "5-year survival rate" not in text
    assert any("don't have to go through this alone" in i.value for i in at.info)
    assert at.text_input(key="onco_result_question")


def test_oncology_non_cancer_and_scan_image(monkeypatch):
    from test_summarizer import ONCO_JSON

    for kind, msg in [("not_cancer_related", "doesn't look like a cancer"), ("scan_image", "scan picture")]:
        at = run(monkeypatch, onco_result={"summary": dict(ONCO_JSON, document_kind=kind), "language": "English"})
        assert any(msg in w.value for w in at.warning)
        assert not any(t.key == "onco_result_question" for t in at.text_input)


def test_oncology_sample_button_uses_careful_mode(monkeypatch):
    import streamlit as st

    import summarizer
    from test_summarizer import ONCO_JSON, ONCO_PDF

    st.cache_resource.clear()
    seen = {}

    def fake_onco(data, language, mode):
        seen.update(data=data, mode=mode)
        return summarizer.OncologySummary.model_validate(ONCO_JSON), 0, []

    monkeypatch.setattr(summarizer, "explain_oncology", fake_onco)
    at = run(monkeypatch)
    at.button(key="onco_sample").click().run()
    assert not at.exception
    assert seen == {"data": ONCO_PDF, "mode": "accurate"}


def test_oncology_followup_passes_no_prognosis(monkeypatch):
    import streamlit as st

    import summarizer
    from test_summarizer import ONCO_JSON

    st.cache_resource.clear()
    seen = {}

    def fake_followup(context, question, history, language, **kwargs):
        seen.update(kwargs)
        return summarizer.FollowUpAnswer(answer="ok", off_topic=False, see_doctor_soon=False)

    monkeypatch.setattr(summarizer, "ask_followup", fake_followup)
    at = run(monkeypatch, onco_result={"summary": ONCO_JSON, "language": "English"})
    at.text_input(key="onco_result_question").input("What is HER2?")
    ask_button(at, "onco").click().run()
    assert seen == {"no_prognosis": True}

import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors
from pypdf import PdfWriter

import summarizer as s

SAMPLES = Path(__file__).resolve().parent.parent / "sample_reports"
PDF = (SAMPLES / "sample_report_2026-03.pdf").read_bytes()
PDF2 = (SAMPLES / "sample_report_2026-09.pdf").read_bytes()

SUMMARY_JSON = {
    "is_lab_report": True,
    "report_date": "12-Mar-2026",
    "lab_name": "Demo Diagnostics Laboratory",
    "overall_summary": "Most results are fine, but a few need attention.",
    "key_findings": ["LDL cholesterol is high."],
    "urgent_flags": [],
    "tests": [
        {"name": "LDL Cholesterol", "category": "Lipid Profile", "value": "151", "unit": "mg/dL",
         "reference_range": "< 100", "status": "high", "explanation": "LDL is the 'bad' cholesterol."},
        {"name": "TSH", "category": "Thyroid", "value": "3.1", "unit": "µIU/mL",
         "reference_range": "0.27 - 4.20", "status": "normal", "explanation": "Your thyroid looks fine."},
    ],
    "questions_for_doctor": ["Should I recheck my cholesterol?"],
    "notes": "",
}

COMPARE_JSON = {
    "both_lab_reports": True,
    "same_person_likely": True,
    "earlier_report_date": "12-Mar-2026",
    "later_report_date": "15-Sep-2026",
    "overall_summary": "Vitamin D improved; LDL went up.",
    "improvements": ["Vitamin D is higher."],
    "concerns": ["LDL rose from 151 to 162."],
    "urgent_flags": [],
    "trends": [
        {"name": "LDL Cholesterol", "unit": "mg/dL", "earlier_value": "151", "later_value": "162",
         "reference_range": "< 100", "change": "worsened", "explanation": "Moved further above range."}
    ],
    "questions_for_doctor": ["What can lower my LDL?"],
    "notes": "",
}


def api_error(code, message="x"):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    """Returns/raises the queued outcomes in order; the last one repeats."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def fake_client(*outcomes):
    return SimpleNamespace(models=FakeModels(*outcomes))


@pytest.fixture(autouse=True)
def no_sleep_and_default_models(monkeypatch):
    sleeps = []
    monkeypatch.setattr(s.time, "sleep", sleeps.append)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("MAX_PDF_PAGES", raising=False)
    return sleeps


# --- request building -------------------------------------------------------


def test_summarize_parses_and_builds_request():
    client = fake_client(json.dumps(SUMMARY_JSON))
    out, corrected = s.summarize_report(PDF, "Hindi", client=client, model="test-model")

    assert isinstance(out, s.ReportSummary)
    assert out.tests[0].status == "high" and corrected == []

    call = client.models.calls[0]
    assert call.model == "test-model"
    pdf_part, prompt = call.contents
    assert pdf_part.inline_data.mime_type == "application/pdf"
    assert pdf_part.inline_data.data == PDF
    assert "Hindi" in prompt
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is s.ReportSummary
    assert "do not diagnose" in call.config.system_instruction
    assert call.config.thinking_config.thinking_level == "LOW"


def test_accurate_mode_uses_high_thinking():
    client = fake_client(json.dumps(SUMMARY_JSON))
    s.summarize_report(PDF, mode="accurate", client=client)
    assert client.models.calls[0].config.thinking_config.thinking_level == "HIGH"


def test_default_model_and_env_override(monkeypatch):
    client = fake_client(json.dumps(SUMMARY_JSON))
    s.summarize_report(PDF, client=client)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    s.summarize_report(PDF, client=client)
    assert [c.model for c in client.models.calls] == [s.DEFAULT_MODEL, "gemini-other"]


def test_compare_sends_reports_in_order():
    client = fake_client(json.dumps(COMPARE_JSON))
    out = s.compare_reports(PDF, PDF2, "Tamil", client=client, model="m")

    assert out.trends[0].change == "worsened"
    contents = client.models.calls[0].contents
    assert contents[0] == "EARLIER REPORT:" and contents[1].inline_data.data == PDF
    assert contents[2] == "LATER REPORT:" and contents[3].inline_data.data == PDF2
    assert "Tamil" in contents[4]
    assert client.models.calls[0].config.response_schema is s.ComparisonSummary


# --- input validation -------------------------------------------------------


def make_pdf(pages: int, password: str | None = None) -> bytes:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(100, 100)
    if password:
        w.encrypt(password)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"hello world", "isn't a PDF or a photo"),
    (b"%PDF" + b"0" * (s.MAX_PDF_BYTES + 1), "larger than 20 MB"),
    (b"%PDF-1.4 garbage", "could not be opened"),
    (make_pdf(16), "has 16 pages. The limit is 15"),
    (make_pdf(1, password="secret"), "password-protected"),
])
def test_validate_pdf_rejects_bad_input(data, msg):
    client = fake_client("{}")
    with pytest.raises(s.SummarizerError, match=msg):
        s.summarize_report(data, client=client)
    assert client.models.calls == []  # nothing sent to Gemini


def test_page_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("MAX_PDF_PAGES", "20")
    s.validate_pdf(make_pdf(16))


# --- retries, fallback, errors ----------------------------------------------


def test_retries_busy_errors_then_succeeds(no_sleep_and_default_models):
    client = fake_client(api_error(503), api_error(503), json.dumps(SUMMARY_JSON))
    out, _ = s.summarize_report(PDF, client=client)
    assert out.is_lab_report
    assert [c.model for c in client.models.calls] == [s.DEFAULT_MODEL] * 3
    assert no_sleep_and_default_models == [2, 5]


def test_falls_back_to_second_model_when_busy():
    client = fake_client(api_error(503), api_error(503), api_error(503), json.dumps(SUMMARY_JSON))
    out, _ = s.summarize_report(PDF, client=client)
    assert out.is_lab_report
    assert [c.model for c in client.models.calls] == [s.DEFAULT_MODEL] * 3 + [s.DEFAULT_FALLBACK_MODEL]


def test_missing_model_goes_straight_to_fallback():
    client = fake_client(api_error(404), json.dumps(SUMMARY_JSON))
    s.summarize_report(PDF, client=client)
    assert [c.model for c in client.models.calls] == [s.DEFAULT_MODEL, s.DEFAULT_FALLBACK_MODEL]


def test_gives_up_with_friendly_busy_message():
    client = fake_client(api_error(503))
    with pytest.raises(s.SummarizerError, match="busy"):
        s.summarize_report(PDF, client=client)
    assert len(client.models.calls) == 4


def test_retries_without_thinking_if_model_rejects_it():
    client = fake_client(api_error(400, "thinking_level is not supported"), json.dumps(SUMMARY_JSON))
    s.summarize_report(PDF, client=client)
    first, second = client.models.calls
    assert first.config.thinking_config is not None and second.config.thinking_config is None


@pytest.mark.parametrize("code,message,expected", [
    (400, "API key not valid. Please pass a valid API key.", "rejected the API key"),
    (403, "x", "rejected the API key"),
    (400, "Request contains an invalid argument.", "couldn't process this request"),
])
def test_non_retryable_errors_are_friendly_and_not_retried(code, message, expected):
    client = fake_client(api_error(code, message))
    with pytest.raises(s.SummarizerError, match=expected) as info:
        s.summarize_report(PDF, client=client)
    assert len(client.models.calls) == 1
    assert "{" not in str(info.value)  # no raw JSON shown to users


def test_rate_limit_message():
    client = fake_client(api_error(429))
    with pytest.raises(s.SummarizerError, match="Too many requests"):
        s.summarize_report(PDF, client=client)


def test_timeout_fails_fast():
    client = fake_client(httpx.ReadTimeout("slow"))
    with pytest.raises(s.SummarizerError, match="took too long"):
        s.summarize_report(PDF, client=client)
    assert len(client.models.calls) == 1


def test_empty_response():
    with pytest.raises(s.SummarizerError, match="empty response"):
        s.summarize_report(PDF, client=fake_client(None))


def test_malformed_response():
    with pytest.raises(s.SummarizerError, match="expected format"):
        s.summarize_report(PDF, client=fake_client('{"is_lab_report": true}'))


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(s.SummarizerError, match="No Gemini API key"):
        s.summarize_report(PDF)


# --- status double-check ----------------------------------------------------


@pytest.mark.parametrize("rng,expected", [
    ("13.0 - 17.0", (13.0, 17.0)),
    ("4000-11000", (4000.0, 11000.0)),
    ("4,000 – 11,000", (4000.0, 11000.0)),
    ("0.27 to 4.20", (0.27, 4.20)),
    ("< 200", (None, 200.0)),
    ("Up to 40", (None, 40.0)),
    ("> 40", (40.0, None)),
    ("M: 13-17, F: 12-15", None),
    ("Negative", None),
    ("", None),
])
def test_parse_range(rng, expected):
    assert s.parse_range(rng) == expected


@pytest.mark.parametrize("value,rng,expected", [
    ("11.8", "13.0 - 17.0", "low"),
    ("232", "< 200", "high"),
    ("38", "> 40", "low"),
    ("7,200", "4000 - 11000", "normal"),
    ("Positive", "Negative", None),
    ("5.2", "M: 4-6, F: 3-5", None),
])
def test_computed_status(value, rng, expected):
    assert s.computed_status(value, rng) == expected


def test_wrong_flags_are_corrected():
    data = json.loads(json.dumps(SUMMARY_JSON))
    data["tests"][0]["status"] = "normal"  # LDL 151 with range < 100 is actually high
    data["tests"][1]["status"] = "high"  # TSH 3.1 in 0.27-4.20 is actually normal
    data["tests"].append({"name": "Glucose", "category": "Diabetes", "value": "99", "unit": "mg/dL",
                          "reference_range": "70 - 100", "status": "borderline", "explanation": "Near the top."})
    out, corrected = s.summarize_report(PDF, client=fake_client(json.dumps(data)))
    assert [t.status for t in out.tests] == ["high", "normal", "borderline"]  # borderline kept
    assert corrected == ["LDL Cholesterol", "TSH"]


# --- photos -----------------------------------------------------------------


def image_bytes(fmt: str) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_photo_reports_are_sent_as_images(fmt, mime):
    client = fake_client(json.dumps(SUMMARY_JSON))
    s.summarize_report(image_bytes(fmt), client=client)
    assert client.models.calls[0].contents[0].inline_data.mime_type == mime


def test_heic_detected_by_content():
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 50
    assert s.detect_mime(heic) == "image/heic"


def test_file_type_comes_from_content_not_name():
    # a text file renamed to .jpg is still rejected
    with pytest.raises(s.SummarizerError, match="isn't a PDF or a photo"):
        s.make_part(b"just some text pretending to be a jpg")


# --- prescriptions / doctor's notes -----------------------------------------

NOTE_JSON = {
    "is_medical_note": True,
    "document_type": "prescription",
    "legibility": "partly_unclear",
    "date": "02/11/26",
    "overall_summary": "The doctor noted a fever and prescribed two medicines.",
    "findings": [{"as_written": "Fever x 3d", "plain_explanation": "Fever for 3 days.", "confidence": "high"}],
    "medicines": [
        {"name_as_written": "Tab Paracetamol", "likely_medicine": "Paracetamol", "strength": "500 mg",
         "form": "tablet", "dose_as_written": "1-0-1 PC", "dose_in_plain_words": "One in the morning and one at night after food.",
         "duration": "x 5d", "usual_purpose": "Fever and pain.", "confidence": "high", "unclear_parts": ""},
        {"name_as_written": "Tab Am??cillin", "likely_medicine": "Amoxicillin", "strength": "",
         "form": "tablet", "dose_as_written": "TDS", "dose_in_plain_words": "Three times a day.",
         "duration": "", "usual_purpose": "Bacterial infections.", "confidence": "low", "unclear_parts": "Middle letters of the name"},
    ],
    "tests_ordered": [{"as_written": "CBC", "plain_explanation": "A blood count test.", "confidence": "high"}],
    "advice": [{"as_written": "Plenty of fluids", "plain_explanation": "Drink lots of water.", "confidence": "high"}],
    "follow_up": "Review after 5 days.",
    "could_not_read": ["Second medicine name"],
    "questions_for_doctor_or_pharmacist": ["What is the second medicine?"],
}


def test_explain_note_request_and_safety():
    client = fake_client(json.dumps(NOTE_JSON))
    photos = [image_bytes("JPEG"), image_bytes("PNG")]
    note, codes = s.explain_note(photos, "Hindi", client=client)

    call = client.models.calls[0]
    assert [c.inline_data.mime_type for c in call.contents[:2]] == ["image/jpeg", "image/png"]
    assert "2 attached images" in call.contents[2] and "Hindi" in call.contents[2]
    assert call.config.system_instruction == s.NOTE_SYSTEM_INSTRUCTION
    assert "Never guess" in call.config.system_instruction
    assert call.config.thinking_config.thinking_level == "HIGH"  # always accurate for handwriting
    assert call.config.response_schema is s.NoteSummary

    clear, unclear = note.medicines
    assert clear.likely_medicine == "Paracetamol" and clear.usual_purpose
    assert unclear.likely_medicine == "" and unclear.usual_purpose == ""  # stripped by code for low confidence
    assert codes["0"] == [("1-0-1", "1 in the morning, none in the afternoon, 1 at night"), ("PC", "after food")]
    assert codes["1"] == [("TDS", "three times a day")]


@pytest.mark.parametrize("n,msg", [(0, "at least one photo"), (6, "at most 5 photos")])
def test_explain_note_photo_count(n, msg):
    with pytest.raises(s.SummarizerError, match=msg):
        s.explain_note([image_bytes("PNG")] * n, client=fake_client("{}"))


def test_explain_note_names_bad_photo():
    with pytest.raises(s.SummarizerError, match="Photo 2 isn't a PDF or a photo"):
        s.explain_note([image_bytes("PNG"), b"nope"], client=fake_client("{}"))


@pytest.mark.parametrize("text,expected", [
    ("1-0-1 PC x 5d", [("1-0-1", "1 in the morning, none in the afternoon, 1 at night"), ("PC", "after food")]),
    ("½ - 0 - ½", [("½-0-½", "½ in the morning, none in the afternoon, ½ at night")]),
    ("bd", [("BD", "twice a day")]),
    ("OD HS", [("OD", "once a day"), ("HS", "at bedtime")]),
    ("12-03-2026", []),  # dates are not dose patterns
    ("Rx gold sachet", []),  # no partial-word matches
])
def test_explain_dose_codes(text, expected):
    assert s.explain_dose_codes(text) == expected


# --- scan and other descriptive reports -------------------------------------

SCAN_PDF = (SAMPLES / "sample_mri_knee.pdf").read_bytes()

SCAN_JSON = {
    "document_kind": "written_report",
    "report_type": "mri",
    "body_part": "Left knee",
    "date": "20-Oct-2026",
    "reason_for_test": "Knee pain for 2 months.",
    "overall_summary": "The MRI shows wear in a knee cartilage pad, a little fluid, and no tears.",
    "main_conclusion": [
        {"as_written": "Grade II degeneration of the medial meniscus. No meniscal tear.",
         "plain_explanation": "The cushion in your knee shows some wear inside, but it is not torn.",
         "status": "needs_attention"},
        {"as_written": "Cruciate and collateral ligaments intact.",
         "plain_explanation": "The main knee ligaments are fine.", "status": "normal"},
    ],
    "findings": [
        {"as_written": "Mild joint effusion is seen.", "plain_explanation": "A little extra fluid in the knee.",
         "status": "needs_attention"},
        {"as_written": "No fracture.", "plain_explanation": "No broken bone.", "status": "normal"},
    ],
    "terms": [{"term": "Joint effusion", "meaning": "Extra fluid inside a joint."}],
    "recommendations": [{"as_written": "Suggested clinical correlation.",
                         "plain_explanation": "Your doctor should compare this with your symptoms.",
                         "status": "normal"}],
    "urgent_flags": [],
    "questions_for_doctor": ["Do I need physiotherapy?"],
    "notes": "",
}


def test_explain_scan_request():
    client = fake_client(json.dumps(SCAN_JSON))
    scan = s.explain_scan(SCAN_PDF, "Kannada", mode="accurate", client=client)
    call = client.models.calls[0]
    assert call.contents[0].inline_data.mime_type == "application/pdf"
    assert "Kannada" in call.contents[1]
    assert call.config.system_instruction == s.SCAN_SYSTEM_INSTRUCTION
    assert "Never look at or interpret medical images" in call.config.system_instruction
    assert call.config.response_schema is s.ScanSummary
    assert call.config.thinking_config.thinking_level == "HIGH"
    assert scan.report_type == "mri" and len(scan.findings) == 2


def test_raw_scan_image_is_never_interpreted():
    # Even if the model ignores its instructions and describes the image, the code throws it away.
    data = dict(SCAN_JSON, document_kind="scan_image")
    scan = s.explain_scan(image_bytes("PNG"), client=fake_client(json.dumps(data)))
    assert scan.document_kind == "scan_image"
    assert scan.findings == [] and scan.main_conclusion == [] and scan.overall_summary == ""
    assert scan.urgent_flags == [] and scan.questions_for_doctor == []


# --- follow-up questions ----------------------------------------------------

FOLLOWUP_JSON = {"answer": "LDL is the 'bad' cholesterol.", "off_topic": False, "see_doctor_soon": False}


def test_followup_uses_stored_result_not_the_file():
    client = fake_client(json.dumps(FOLLOWUP_JSON))
    history = [{"q": f"q{i}", "a": f"a{i}"} for i in range(5)]
    ans = s.ask_followup(SUMMARY_JSON, "  What is LDL?  ", history, "Hindi", client=client)
    assert ans.answer.startswith("LDL")

    call = client.models.calls[0]
    assert all(isinstance(c, str) for c in call.contents)  # no file re-uploaded
    assert '"LDL Cholesterol"' in call.contents[0]
    assert "q2" in call.contents[1] and "q4" in call.contents[1] and "q1" not in call.contents[1]  # last 3 turns
    assert call.contents[2].startswith("QUESTION: What is LDL?") and "Hindi" in call.contents[2]
    assert call.config.system_instruction == s.FOLLOWUP_SYSTEM_INSTRUCTION
    assert "Never recommend, change, add or stop medicines" in call.config.system_instruction


def test_followup_without_history():
    client = fake_client(json.dumps(FOLLOWUP_JSON))
    s.ask_followup(SUMMARY_JSON, "What is LDL?", client=client)
    assert len(client.models.calls[0].contents) == 2


@pytest.mark.parametrize("q,msg", [("   ", "type a question"), ("x" * 501, "under 500 characters")])
def test_followup_input_checks(q, msg):
    client = fake_client("{}")
    with pytest.raises(s.SummarizerError, match=msg):
        s.ask_followup(SUMMARY_JSON, q, client=client)
    assert client.models.calls == []


# --- oncology ---------------------------------------------------------------

ONCO_PDF = (SAMPLES / "sample_breast_biopsy.pdf").read_bytes()

ONCO_JSON = {
    "document_kind": "pathology_report",
    "cancer_type_as_written": "Invasive ductal carcinoma (NST)",
    "organ": "Left breast",
    "date": "28-Oct-2026",
    "overall_summary": "The report confirms a breast cancer that was removed with clear margins. "
                       "The 5-year survival rate for this stage is high. One lymph node had cancer cells.",
    "key_points": [{"as_written": "Discuss at multidisciplinary tumour board.",
                    "plain_explanation": "A team of specialists will review your case together."}],
    "staging": {"tnm_as_written": "pT2 pN1a", "stage_group_as_written": "", "grade_as_written": "Grade 2"},
    "tumour_size": {"as_written": "2.4 cm", "plain_explanation": "About the size of a grape."},
    "margins": {"as_written": "All margins free of tumour", "plain_explanation": "No cancer at the edges of the removed tissue."},
    "lymph_nodes": {"as_written": "1 of 12 positive", "plain_explanation": "Cancer cells were found in 1 of 12 nodes."},
    "biomarkers": [
        {"name": "Estrogen receptor (ER)", "result_as_written": "Positive (Allred 8/8)",
         "plain_explanation": "The cancer uses estrogen to grow. This is usually curable."},
        {"name": "HER2/neu", "result_as_written": "Negative (IHC 1+)", "plain_explanation": "HER2 is not driving growth."},
    ],
    "tumour_markers": [
        {"name": "CA 15-3", "value": "42", "unit": "U/mL", "reference_range": "< 30", "status": "normal",
         "plain_explanation": "Slightly above the usual range."},
    ],
    "terms": [{"as_written": "Margins", "plain_explanation": "The edges of the removed tissue."}],
    "questions_for_doctor": ["What does my prognosis look like?", "Will I need chemotherapy?"],
    "notes": "",
}


def test_explain_oncology_request_and_safety():
    client = fake_client(json.dumps(ONCO_JSON))
    onco, removed, corrected = s.explain_oncology(ONCO_PDF, "Hindi", client=client)

    call = client.models.calls[0]
    assert call.contents[0].inline_data.mime_type == "application/pdf"
    assert "Hindi" in call.contents[1]
    assert call.config.system_instruction == s.ONCOLOGY_SYSTEM_INSTRUCTION
    assert "Never give survival rates" in call.config.system_instruction
    assert call.config.response_schema is s.OncologySummary
    assert call.config.thinking_config.thinking_level == "HIGH"  # careful mode by default

    # prognosis sentences removed by code, the rest kept
    assert "survival" not in onco.overall_summary
    assert onco.overall_summary.startswith("The report confirms") and "One lymph node" in onco.overall_summary
    assert onco.biomarkers[0].plain_explanation == "The cancer uses estrogen to grow."
    assert removed == 2
    # questions for the oncologist may mention prognosis: that's the right person to ask
    assert onco.questions_for_doctor[0] == "What does my prognosis look like?"
    # tumour marker flag recomputed from the printed range
    assert onco.tumour_markers[0].status == "high" and corrected == ["CA 15-3"]


def test_oncology_scan_image_is_wiped():
    data = dict(ONCO_JSON, document_kind="scan_image")
    onco, removed, corrected = s.explain_oncology(image_bytes("JPEG"), client=fake_client(json.dumps(data)))
    assert onco.document_kind == "scan_image"
    assert onco.overall_summary == "" and onco.biomarkers == [] and onco.staging.tnm_as_written == ""
    assert onco.cancer_type_as_written == "" and onco.questions_for_doctor == []


@pytest.mark.parametrize("text,codes", [
    ("pT2 pN1a M0", ["pT2", "pN1a", "M0"]),
    ("T2N1M0", ["T2", "N1", "M0"]),
    ("ypT1c N0 cM0", ["ypT1c", "N0", "cM0"]),
    ("pTis pNX", ["pTis", "pNX"]),
    ("Tumour 2 cm, T2W images", []),
])
def test_explain_tnm(text, codes):
    assert [c for c, _ in s.explain_tnm(text)] == codes


def test_tnm_meanings_from_fixed_glossary():
    meanings = dict(s.explain_tnm("ypT0 N0 M1b"))
    assert meanings["ypT0"].startswith("no sign of a main tumour (measured after treatment")
    assert meanings["N0"] == "no cancer found in nearby lymph nodes"
    assert meanings["M1b"].startswith("the cancer has spread")


@pytest.mark.parametrize("text,label", [
    ("Stage IIB", "Stage IIB"), ("stage 3", "Stage III"), ("AJCC Stage IV", "Stage IV"), ("none", None),
])
def test_explain_stage(text, label):
    out = s.explain_stage(text)
    assert (out[0] if out else None) == label


@pytest.mark.parametrize("text,starts", [
    ("Grade 2 (Nottingham 6/9)", "intermediate grade"), ("G3", "high grade"),
    ("Poorly differentiated", "high grade"), ("low-grade", "low grade"), ("Gleason 7", None),
])
def test_explain_grade(text, starts):
    out = s.explain_grade(text)
    assert (out[1].split(":")[0] if out else None) == starts


@pytest.mark.parametrize("name,key", [
    ("Estrogen receptor (ER)", "ER"), ("Progesterone receptor", "PR"), ("HER2/neu", "HER2"),
    ("Ki67 index", "KI-67"), ("ERBB2 amplification", None), ("MSI-H", "MSI"), ("BRCA1", "BRCA"),
    ("CA 125", "CA-125"), ("CA19-9", "CA 19-9"), ("Carcinoembryonic antigen", "CEA"),
])
def test_glossary_matching(name, key):
    glossary = {**s.BIOMARKER_GLOSSARY, **s.TUMOUR_MARKER_GLOSSARY}
    from medexplain.oncology import _glossary_key

    assert _glossary_key(name, glossary) == key


@pytest.mark.parametrize("text,kept,removed", [
    ("It is small. Survival is good. Ask your doctor.", "It is small. Ask your doctor.", 1),
    ("कैंसर छोटा है। इसका prognosis अच्छा है। डॉक्टर से पूछें।", "कैंसर छोटा है। डॉक्टर से पूछें।", 1),
    ("Nothing to remove here.", "Nothing to remove here.", 0),
])
def test_strip_prognosis(text, kept, removed):
    assert s.strip_prognosis(text) == (kept, removed)


def test_oncology_followup_never_gives_outlook():
    reply = dict(FOLLOWUP_JSON, answer="HER2 negative means HER2 is not driving growth. The cure rate is about 90%.")
    ans = s.ask_followup(ONCO_JSON, "Will I be cured?", no_prognosis=True, client=fake_client(json.dumps(reply)))
    assert "90%" not in ans.answer
    assert ans.answer.startswith("HER2 negative means") and s.OUTLOOK_MESSAGE in ans.answer
    assert "Never give survival rates" in s.FOLLOWUP_SYSTEM_INSTRUCTION


@pytest.mark.parametrize("sentence", [
    "The cure rate is about 90%.", "Most women are cured.", "The recurrence rate is low.",
    "Your chance of relapse is small.", "Life expectancy is normal.", "This stage is rarely fatal.",
])
def test_prognosis_phrasings_are_caught(sentence):
    assert s.strip_prognosis(sentence) == ("", 1)


@pytest.mark.parametrize("sentence", [
    "Discuss the treatment plan with your oncologist.", "The margins are clear.", "Ki-67 is 18%.",
])
def test_normal_sentences_are_kept(sentence):
    assert s.strip_prognosis(sentence) == (sentence, 0)

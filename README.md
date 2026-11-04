# 🧪 Lab Report Explainer

Upload a lab report, a scan report (X-ray, CT, MRI, ultrasound) or a photo of a handwritten prescription, and get a plain-language explanation from Google Gemini, in English, Hindi, Kannada or 12 other languages. Then ask follow-up questions about it.

**Live demo:** _add your Streamlit app link here_. Click **Try a sample report** to see it work without uploading anything.

> ⚠️ This is an educational tool, not medical advice. Always discuss results with a doctor.

## Features

- **Lab reports**: overall summary, key findings, every test with a 🟢/🔴/🔵 status, a plain-language note per test, and questions to ask your doctor
- **Compare two reports**: side-by-side values, what improved, what needs attention
- **Scans & other reports**: written X-ray, CT, MRI, ultrasound, ECG, echo and biopsy reports explained: main conclusion, each finding marked normal or needs attention, medical words explained, and what the report recommends
- **Prescriptions & doctor's notes**: photograph a handwritten note (up to 5 pages); medicines, doses, tests and advice explained, with a confidence rating per item
- **Follow-up questions**: ask up to 5 questions about any result ("what does joint effusion mean?"). Answers are based on the stored explanation, so the file isn't uploaded again
- **15 languages**: explanations in the chosen language's script; test names and values stay exactly as printed
- **Photos and scanned PDFs**: JPG, PNG, WEBP and HEIC, checked by file content rather than file name; Gemini reads PDFs directly, so there's no separate OCR step

### Safety and reliability

- **Code checks the AI**: Python re-checks every high/low/normal flag against the printed reference range and corrects the AI if it got one wrong
- **Prescription safety**: always uses Accurate mode; unclear medicine names are never guessed (the code removes any interpretation the AI marks as unclear); dose shorthand like `1-0-1`, `BD`, `TDS`, `PC` is explained from a fixed table, not by the AI
- **Scan safety**: only the written report is explained. If someone uploads the scan picture itself, the app refuses and the code discards anything the AI said about it
- **Follow-up safety**: no diagnoses, no medicine advice, off-topic questions declined, and possible emergencies (e.g. chest pain) get a clear "seek help now" message
- **Prompt-injection resistant**: documents and questions are treated as data; instructions hidden inside them are ignored
- **Handles busy servers**: retries busy/overloaded errors twice, then switches to a backup model
- **Abuse limits**: 10 requests per visitor per hour, 60 per hour for the whole app, 15 pages per PDF, 20 MB per file, 90-second timeout
- **Nothing stored**: files are processed in memory; results live only in the browser session

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your key in .env
```

Get a Gemini API key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY` in `.env`.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501 and click a **Try a sample…** button, or upload your own. Everything in `sample_reports/` is synthetic (fake patient, fake lab).

## Project layout

```
app.py                   Streamlit UI (tabs, tables, follow-up chat, usage limits)
summarizer.py            Gemini calls, JSON schemas, prompts, safety checks, error handling
scripts/make_samples.py  Regenerates the synthetic sample PDFs (needs reportlab)
sample_reports/          Two fake lab reports (March and September) and a fake knee MRI report
tests/                   pytest suite, 83 tests (Gemini is mocked, no key needed)
```

## How it works

1. The file is validated (real PDF or photo by its first bytes, size, page count, not password-protected).
2. It goes to Gemini with a task-specific system prompt (rules) and a `response_schema`, which forces Gemini to return JSON matching the Pydantic models in `summarizer.py` (`ReportSummary`, `ComparisonSummary`, `ScanSummary`, `NoteSummary`).
3. The JSON is validated, then checked by plain code (range checks, dose-code table, unclear-item and scan-image removal).
4. Streamlit renders the result; follow-up questions send the stored JSON, not the file.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model that accepts PDFs, e.g. `gemini-3.5-flash` (longer support window) or `gemini-3.5-flash-lite` (cheaper) |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash` | Used when the main model stays busy or isn't available |
| `MAX_PDF_PAGES` | `15` | Larger PDFs are rejected before anything is sent to Gemini |
| `SESSION_LIMIT_PER_HOUR` | `10` | Requests (reports and follow-up questions) per visitor per hour |
| `GLOBAL_LIMIT_PER_HOUR` | `60` | Requests per hour for the whole app (protects your API quota) |
| `DEMO_MODE` | off | Set to `true` on a public demo: shows a "don't upload real reports" banner |

## Tests

```bash
pip install pytest
pytest
```

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub (`.env` is git-ignored, so your key stays local).
2. At https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**, pick the repo, branch `main`, file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 and paste this into **Secrets** (with your real key):
   ```toml
   GEMINI_API_KEY = "your-key-here"
   DEMO_MODE = "true"
   ```
4. Deploy. For a private app instead, deploy from a private repo and invite viewers with **Share**.

## Privacy notes

- Files are sent to Google's Gemini API. On the free tier, Google may use inputs to improve its products, which is why the public demo shows a "don't upload real reports" banner. Use a paid (billing-enabled) key if you handle real patient data.
- The app writes nothing to disk and never logs report contents.
- The usage limits live in memory, so they reset when the app restarts. They stop casual abuse, not a determined attacker. Use a separate API key for the public demo so it can be revoked without affecting anything else.

## Ideas for next steps

- An accuracy evaluation on a labelled dataset (e.g. handwritten medicine names) with a published score
- Download the summary as a PDF
- Charts of values over time across 3+ reports
- Read the summary aloud in the chosen language

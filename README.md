# 🧪 Lab Report Explainer

Upload a lab report (PDF or photo) and get a plain-language explanation from Google Gemini — in English, Hindi or 13 other languages. You can also compare two reports, or photograph a handwritten prescription or doctor's note.

> ⚠️ This is an educational tool, not medical advice. Always discuss results with a doctor.

## Features

- **Explain a report** — overall summary, key findings, every test with a 🟢/🔴/🔵 status, a plain-language note per test, and questions to ask your doctor
- **Compare two reports** — side-by-side values, what improved, what needs attention
- **Prescriptions & doctor's notes** — photograph a handwritten note (up to 5 pages); medicines, doses, tests and advice explained, with a confidence rating per item
- **Prescription safety** — always uses Accurate mode; never guesses unclear medicine names (the code hides interpretations the AI marks as unclear); dose shorthand like `1-0-1`, `BD`, `TDS`, `PC` is explained from a fixed table, not by the AI; every result tells the user to confirm with a pharmacist
- **Photos work everywhere** — JPG, PNG, WEBP and HEIC, checked by file content rather than file name
- **15 languages** — explanations in the chosen language's script; test names and values stay exactly as printed
- **Scanned PDFs work** — Gemini reads the PDF directly, so there's no separate OCR step
- **Urgent flags** — clearly dangerous values are highlighted with advice to contact a doctor
- **Safety prompt** — explains without diagnosing, never suggests medicines, never invents values
- **Nothing stored** — files are processed in memory; results live only in your browser session
- **Handles busy servers** — retries busy/overloaded errors twice, then switches to a backup model
- **Fast or Accurate mode** — low thinking by default; Accurate mode (sidebar) thinks longer for messy scans
- **Flag double-check** — Python re-checks each high/low/normal flag against the printed reference range and corrects the AI if it got one wrong
- **Abuse limits** — max 10 reports per visitor per hour and 60 per hour for the whole app, 15 pages per PDF, 20 MB per file, 90-second timeout

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

Open http://localhost:8501 and click **Try a sample report**, or upload your own. The reports in `sample_reports/` are synthetic (fake patient, fake lab).

## Project layout

```
app.py                 Streamlit UI (tabs, tables, rendering)
summarizer.py          Gemini calls, JSON schemas, prompts, error handling
scripts/make_samples.py  Regenerates the synthetic sample PDFs (needs reportlab)
sample_reports/        Two fake lab reports (March and September) for testing
tests/                 pytest suite (Gemini is mocked, no key needed)
```

## How it works

1. The PDF bytes go to Gemini as an inline `application/pdf` part (up to 20 MB).
2. `response_schema` forces Gemini to return JSON matching the Pydantic models in `summarizer.py` (`ReportSummary`, `ComparisonSummary`).
3. The JSON is validated and rendered in Streamlit.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model that accepts PDFs, e.g. `gemini-3.5-flash` (longer support window) or `gemini-3.5-flash-lite` (cheaper) |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash` | Used when the main model stays busy or isn't available |
| `MAX_PDF_PAGES` | `15` | Larger PDFs are rejected before anything is sent to Gemini |
| `SESSION_LIMIT_PER_HOUR` | `10` | Reports per visitor per hour |
| `GLOBAL_LIMIT_PER_HOUR` | `60` | Reports per hour for the whole app (protects your API bill) |
| `DEMO_MODE` | off | Set to `true` on a public demo: shows a "don't upload real reports" banner |

## Tests

```bash
pip install pytest
pytest
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a **private** GitHub repo (`.env` is git-ignored, so your key stays local).
2. At https://share.streamlit.io choose **Create app → Deploy from GitHub**, pick the repo, branch `main`, file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 and paste the contents of `.streamlit/secrets.toml.example` (with your real key) into **Secrets**.
4. Deploy, then use **Share** to invite viewers by email.

## Privacy notes

- PDFs are sent to Google's Gemini API. On the free tier, Google may use inputs to improve its products; use a paid (billing-enabled) key if you handle real patient data.
- The app writes nothing to disk and never logs report contents. Don't deploy it publicly without adding authentication; keep the Streamlit app private and invite viewers.
- Set a budget alert for your Gemini key in Google Cloud Billing so unexpected usage can't surprise you.
- The usage limits live in memory, so they reset when the app restarts. They stop casual abuse, not a determined attacker; the private-app setting is the main protection.

## Ideas for next steps

- Download the summary as a PDF
- Charts of values over time across 3+ reports
- Deploy to Streamlit Community Cloud (put the key in the app's Secrets)

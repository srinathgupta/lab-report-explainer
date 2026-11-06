"""Shared building blocks: settings, file checks, and the Gemini call with retries and fallback."""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Literal, Optional

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

log = logging.getLogger("labreport")

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash"
MAX_PDF_BYTES = 20 * 1024 * 1024  # inline request limit for Gemini is ~20 MB
REQUEST_TIMEOUT_S = 90
RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRY_DELAYS_S = (2, 5)  # waits between attempts on the main model; then one try on the fallback model

Mode = Literal["fast", "accurate"]
THINKING_LEVELS = {"fast": "LOW", "accurate": "HIGH"}

LANGUAGES = [
    "English",
    "Hindi",
    "Bengali",
    "Marathi",
    "Telugu",
    "Tamil",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
    "Odia",
    "Urdu",
    "Spanish",
    "French",
    "Arabic",
]


def max_pages() -> int:
    return int(os.environ.get("MAX_PDF_PAGES", "15"))


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


class SummarizerError(Exception):
    """An error whose message is safe and friendly to show to the user."""


def get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or get_api_key()
    if not key:
        raise SummarizerError("No Gemini API key found. Set GEMINI_API_KEY in your .env file.")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))


def validate_pdf(data: bytes, label: str = "File") -> None:
    if not data:
        raise SummarizerError(f"{label} is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise SummarizerError(f"{label} is larger than 20 MB. Please upload a smaller PDF.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise SummarizerError(f"{label} doesn't look like a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise SummarizerError(f"{label} is password-protected. Please upload an unlocked PDF.")
        pages = len(reader.pages)
    except SummarizerError:
        raise
    except Exception as e:
        log.warning("Could not read PDF: %s", e)
        raise SummarizerError(f"{label} could not be opened. It may be damaged; try downloading it again.") from e
    if pages > max_pages():
        raise SummarizerError(
            f"{label} has {pages} pages. The limit is {max_pages()}; please upload only the pages with your results."
        )


UPLOAD_TYPES = ["pdf", "jpg", "jpeg", "png", "webp", "heic", "heif"]


def detect_mime(data: bytes) -> Optional[str]:
    """Identify the file from its first bytes, not its name (names can lie)."""
    head = data[:16]
    if data.lstrip()[:5].startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
        return "image/heic"
    return None


def make_part(data: bytes, label: str = "File") -> types.Part:
    """Validate an uploaded PDF or photo and wrap it for Gemini."""
    if not data:
        raise SummarizerError(f"{label} is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise SummarizerError(f"{label} is larger than 20 MB. Please upload a smaller file.")
    mime = detect_mime(data)
    if mime is None:
        raise SummarizerError(f"{label} isn't a PDF or a photo (JPG, PNG, WEBP or HEIC).")
    if mime == "application/pdf":
        validate_pdf(data, label)
    return types.Part.from_bytes(data=data, mime_type=mime)


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------


def _friendly(e: errors.APIError, model: str) -> SummarizerError:
    text = str(e)
    code = getattr(e, "code", None)
    if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
        return SummarizerError("Gemini rejected the API key. Check GEMINI_API_KEY and restart the app.")
    if code == 429:
        return SummarizerError("Too many requests to Gemini right now. Please wait a minute and try again.")
    if code == 404:
        return SummarizerError(f"Model '{model}' isn't available. Set GEMINI_MODEL to an available model.")
    if code in RETRYABLE_CODES:
        return SummarizerError("Gemini is busy right now. Please try again in a minute.")
    return SummarizerError("Gemini couldn't process this request. Please try again, or try a different file.")


def _config(schema: type[BaseModel], mode: Mode, with_thinking: bool, system: str) -> types.GenerateContentConfig:
    kwargs = dict(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.2,
    )
    if with_thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVELS[mode])
    return types.GenerateContentConfig(**kwargs)


def _call_once(client, model: str, contents: list, schema, mode: Mode, system: str):
    """One request. If the model rejects the thinking setting, retry once without it."""
    try:
        return client.models.generate_content(
            model=model, contents=contents, config=_config(schema, mode, True, system)
        )
    except errors.ClientError as e:
        if getattr(e, "code", None) == 400 and "thinking" in str(e).lower():
            log.warning("Model %s rejected thinking_level; retrying without it", model)
            return client.models.generate_content(
                model=model, contents=contents, config=_config(schema, mode, False, system)
            )
        raise


def _generate(
    client, model: str, contents: list, schema: type[BaseModel], mode: Mode, system: str
) -> BaseModel:
    """Try the main model up to 3 times (busy/rate-limit errors), then the fallback model once."""
    fallback = os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    attempts = [model] * (len(RETRY_DELAYS_S) + 1)
    if fallback and fallback != model:
        attempts.append(fallback)

    response = None
    last_error: Optional[SummarizerError] = None
    i = 0
    while i < len(attempts):
        attempt_model = attempts[i]
        try:
            response = _call_once(client, attempt_model, contents, schema, mode, system)
            break
        except errors.APIError as e:
            code = getattr(e, "code", None)
            log.warning("Gemini error on %s (attempt %d): %s", attempt_model, i + 1, e)
            last_error = _friendly(e, attempt_model)
            if code == 404 and attempt_model != attempts[-1]:
                i = len(attempts) - 1  # main model missing: go straight to the fallback
                continue
            if code not in RETRYABLE_CODES:
                raise last_error from e
        except httpx.TimeoutException as e:
            log.warning("Gemini timeout on %s", attempt_model)
            raise SummarizerError(
                "Gemini took too long to answer. Please try again, or turn off Accurate mode."
            ) from e
        if i < len(RETRY_DELAYS_S):
            time.sleep(RETRY_DELAYS_S[i])
        i += 1
    if response is None:
        raise last_error or SummarizerError("Gemini couldn't process this request. Please try again.")

    text = getattr(response, "text", None)
    if not text:
        raise SummarizerError("Gemini returned an empty response (it may have been blocked). Please try again.")
    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        log.warning("Unexpected response format: %s", e)
        raise SummarizerError("Gemini's response wasn't in the expected format. Please try again.") from e

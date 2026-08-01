"""Resolves every image and voice note into structured data once, up front,
and caches results in media_cache.json — the main pipeline only ever reads
this cache, it never calls the vision API or Whisper per-message."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_PATH = Path(__file__).resolve().parent / "media_cache.json"

VISION_MODEL = "gemini-3.6-flash"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL_SIZE = "base"

# Free-tier RPM/RPD quotas are scoped per-model-per-project, not per-key, so
# rotating through several Gemini models on a 429 gets a fresh quota bucket
# instead of failing straight to Groq. Confirmed available on this account
# via client.models.list().
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
]

# Lowered from 5: the free tier's advertised 5/min cap was observed being hit
# at 6 in-flight (timing edge at the window boundary), so 4 leaves margin.
GEMINI_RATE_LIMIT_CALLS = 4
GEMINI_RATE_LIMIT_PERIOD_SECONDS = 60.0

# Groq's free tier is generally generous (~30 requests/minute for most
# models) but isn't published per-model, so this is a conservative default.
GROQ_RATE_LIMIT_CALLS = 30
GROQ_RATE_LIMIT_PERIOD_SECONDS = 60.0

load_dotenv(REPO_ROOT / ".env")


class RateLimiter:
    """Sliding-window limiter: wait() blocks until a call is allowed under
    max_calls per period_seconds. Thread-safe so a single instance can be
    shared across callers."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._evict(time.monotonic())
            self._calls.append(time.monotonic())

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.period_seconds:
            self._calls.popleft()


gemini_rate_limiter = RateLimiter(GEMINI_RATE_LIMIT_CALLS, GEMINI_RATE_LIMIT_PERIOD_SECONDS)
groq_rate_limiter = RateLimiter(GROQ_RATE_LIMIT_CALLS, GROQ_RATE_LIMIT_PERIOD_SECONDS)

# One limiter per Gemini model, since each model has its own independent
# RPM/RPD quota bucket on the free tier.
_gemini_model_rate_limiters = {
    model: RateLimiter(GEMINI_RATE_LIMIT_CALLS, GEMINI_RATE_LIMIT_PERIOD_SECONDS)
    for model in GEMINI_MODELS
}


def _rate_limiter_for_model(model: str) -> RateLimiter:
    if model not in _gemini_model_rate_limiters:
        _gemini_model_rate_limiters[model] = RateLimiter(
            GEMINI_RATE_LIMIT_CALLS, GEMINI_RATE_LIMIT_PERIOD_SECONDS
        )
    return _gemini_model_rate_limiters[model]


def _is_transient_gemini_error(exc) -> bool:
    from google.genai import errors as genai_errors

    if not isinstance(exc, (genai_errors.ClientError, genai_errors.ServerError)):
        return False
    code = getattr(exc, "code", None) or 0
    return code == 429 or code >= 500


def call_gemini_with_model_rotation(client, build_kwargs, models=None):
    """Tries each model in `models` (default GEMINI_MODELS) in order, rate
    limited per model. On a transient 429/5xx, moves to the next model
    instead of failing straight out to the Groq fallback — each model has
    its own quota bucket, so a 429 on one doesn't mean Gemini is exhausted.

    `build_kwargs(model_name)` must return the kwargs dict for
    client.models.generate_content(). Returns (response, model_used).
    Raises the last exception if every model in the rotation fails."""
    from google.genai import errors as genai_errors

    models = models or GEMINI_MODELS
    last_exc = None
    for model in models:
        _rate_limiter_for_model(model).wait()
        try:
            response = client.models.generate_content(**build_kwargs(model))
            return response, model
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            if not _is_transient_gemini_error(exc):
                raise
            print(f"    [{exc.code}] {model} unavailable, trying next Gemini model...")
            last_exc = exc
            continue
    raise last_exc

_IMAGE_PROMPT = """This image was sent as a WhatsApp message attachment. Analyze it and \
respond with ONLY a JSON object (no markdown fences, no other text) with these keys:

- "ocr_text": all readable text in the image, transcribed verbatim. Empty string if none.
- "poster_type": one of "sale", "event", "payment_request", "official_notice", \
"random_forward", "personal_photo", "other".
- "risk_flags": array of zero or more of "urgency_language", "qr_or_payment_prompt", \
"suspicious_link", "brand_impersonation". Empty array if none apply.
"""


def load_media_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_media_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_gemini_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env in the repo "
            "root and fill in your key, or set it as an environment variable."
        )
    return genai.Client(api_key=api_key)


def get_groq_client():
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env in the repo "
            "root and fill in your key, or set it as an environment variable."
        )
    return Groq(api_key=api_key)


def _parse_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "ocr_text": "",
            "poster_type": "other",
            "risk_flags": [],
            "_parse_error": raw_text,
        }


def resolve_image(image_id: str, file_path: str, client=None) -> dict:
    from google.genai import types

    client = client or get_gemini_client()
    full_path = DATASET_DIR / file_path
    media_type = mimetypes.guess_type(full_path.name)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=full_path.read_bytes(), mime_type=media_type)

    response, model_used = call_gemini_with_model_rotation(
        client, lambda model: dict(model=model, contents=[image_part, _IMAGE_PROMPT])
    )
    parsed = _parse_json_response(response.text)

    return {
        "media_id": image_id,
        "media_type": "image",
        "file_path": file_path,
        "ocr_text": parsed.get("ocr_text", ""),
        "poster_type": parsed.get("poster_type", "other"),
        "risk_flags": parsed.get("risk_flags", []),
        "resolved_by": "gemini",
        "model": model_used,
    }


def resolve_image_via_groq(image_id: str, file_path: str, client=None) -> dict:
    """Fallback path used when Gemini is rate-limited/unavailable. Same
    prompt and output shape as resolve_image(), different provider."""
    client = client or get_groq_client()
    full_path = DATASET_DIR / file_path
    media_type = mimetypes.guess_type(full_path.name)[0] or "image/jpeg"
    encoded = base64.standard_b64encode(full_path.read_bytes()).decode("utf-8")

    groq_rate_limiter.wait()
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _IMAGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    },
                ],
            }
        ],
    )
    parsed = _parse_json_response(response.choices[0].message.content)

    return {
        "media_id": image_id,
        "media_type": "image",
        "file_path": file_path,
        "ocr_text": parsed.get("ocr_text", ""),
        "poster_type": parsed.get("poster_type", "other"),
        "risk_flags": parsed.get("risk_flags", []),
        "resolved_by": "groq",
        "model": GROQ_VISION_MODEL,
    }


_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def resolve_voice(voice_note_id: str, file_path: str) -> dict:
    model = _get_whisper_model()
    full_path = DATASET_DIR / file_path
    result = model.transcribe(str(full_path))

    return {
        "media_id": voice_note_id,
        "media_type": "voice",
        "file_path": file_path,
        "transcript": result["text"].strip(),
    }


def get_media(media_id: str, cache: Optional[dict] = None) -> Optional[dict]:
    cache = cache if cache is not None else load_media_cache()
    return cache.get(media_id)

"""Resolves every image and voice note into structured data once, up front,
and caches results in media_cache.json — the main pipeline only ever reads
this cache, it never calls the vision API or Whisper per-message."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_PATH = Path(__file__).resolve().parent / "media_cache.json"

VISION_MODEL = "claude-sonnet-5"
WHISPER_MODEL_SIZE = "base"

load_dotenv(REPO_ROOT / ".env")

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


def get_anthropic_client():
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env in the repo "
            "root and fill in your key, or set it as an environment variable."
        )
    return Anthropic(api_key=api_key)


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
    client = client or get_anthropic_client()
    full_path = DATASET_DIR / file_path
    media_type = mimetypes.guess_type(full_path.name)[0] or "image/jpeg"
    encoded = base64.standard_b64encode(full_path.read_bytes()).decode("utf-8")

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": _IMAGE_PROMPT},
                ],
            }
        ],
    )
    parsed = _parse_json_response(response.content[0].text)

    return {
        "media_id": image_id,
        "media_type": "image",
        "file_path": file_path,
        "ocr_text": parsed.get("ocr_text", ""),
        "poster_type": parsed.get("poster_type", "other"),
        "risk_flags": parsed.get("risk_flags", []),
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

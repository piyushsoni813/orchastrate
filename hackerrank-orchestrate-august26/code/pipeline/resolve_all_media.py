"""One-time script: resolves every image in images.csv (Gemini vision,
falling back to Groq if Gemini is rate-limited/unavailable) and every voice
note in voice_notes.csv (local Whisper) into media_cache.json.

Run with: python code/pipeline/resolve_all_media.py

Safe to re-run: already-cached media_ids are skipped, and the cache is saved
after every item so a failed/interrupted run doesn't lose prior progress.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from google.genai import errors as genai_errors

from pipeline.media_resolver import (
    CACHE_PATH,
    DATASET_DIR,
    get_gemini_client,
    get_groq_client,
    load_media_cache,
    resolve_image,
    resolve_image_via_groq,
    resolve_voice,
    save_media_cache,
)

# Per-minute pacing against gemini-2.5-flash's free-tier limit lives in
# media_resolver.gemini_rate_limiter, shared by every resolve_image() call.

_groq_client = None


def _get_groq_client_lazy():
    global _groq_client
    if _groq_client is None:
        _groq_client = get_groq_client()
    return _groq_client


def _resolve_image_with_fallback(image_id: str, file_path: str, gemini_client) -> Optional[dict]:
    try:
        return resolve_image(image_id, file_path, client=gemini_client)
    except (genai_errors.ClientError, genai_errors.ServerError) as exc:
        is_transient = exc.code == 429 or exc.code >= 500
        if not is_transient:
            raise
        print(f"  [{exc.code}] {image_id}: Gemini unavailable ({exc.message}), "
              f"falling back to Groq ...")

    try:
        return resolve_image_via_groq(image_id, file_path, client=_get_groq_client_lazy())
    except Exception as exc:  # any Groq failure (bad model id, quota, auth) should not kill the batch
        print(f"  {image_id}: Groq fallback also failed ({exc}); leaving unresolved "
              f"for a later retry.")
        return None


def main() -> None:
    images = pd.read_csv(DATASET_DIR / "images.csv")
    voice_notes = pd.read_csv(DATASET_DIR / "voice_notes.csv")
    cache = load_media_cache()

    client = get_gemini_client()
    for _, row in images.iterrows():
        image_id = row["image_id"]
        if image_id in cache:
            print(f"[skip] {image_id} already cached")
            continue
        print(f"[image] resolving {image_id} ...")
        result = _resolve_image_with_fallback(image_id, row["file_path"], client)
        if result is None:
            continue
        cache[image_id] = result
        save_media_cache(cache)

    for _, row in voice_notes.iterrows():
        voice_id = row["voice_note_id"]
        if voice_id in cache:
            print(f"[skip] {voice_id} already cached")
            continue
        print(f"[voice] transcribing {voice_id} ...")
        cache[voice_id] = resolve_voice(voice_id, row["file_path"])
        save_media_cache(cache)

    print(f"\nDone. {len(cache)} entries written to {CACHE_PATH}")
    print("\n=== Resolved media (spot-check) ===\n")
    for media_id, entry in cache.items():
        if entry["media_type"] == "image":
            print(f"{media_id} [{entry['poster_type']}] (via {entry.get('resolved_by', '?')}) "
                  f"risk={entry['risk_flags']}")
            print(f"    ocr_text: {entry['ocr_text'][:100]!r}")
        else:
            print(f"{media_id} [voice] transcript: {entry['transcript'][:100]!r}")


if __name__ == "__main__":
    main()

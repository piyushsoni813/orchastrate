"""One-time script: resolves every image in images.csv (Claude vision) and
every voice note in voice_notes.csv (local Whisper) into media_cache.json.

Run with: python code/pipeline/resolve_all_media.py

Safe to re-run: already-cached media_ids are skipped, and the cache is saved
after every item so a failed/interrupted run doesn't lose prior progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.media_resolver import (
    CACHE_PATH,
    DATASET_DIR,
    get_anthropic_client,
    load_media_cache,
    resolve_image,
    resolve_voice,
    save_media_cache,
)


def main() -> None:
    images = pd.read_csv(DATASET_DIR / "images.csv")
    voice_notes = pd.read_csv(DATASET_DIR / "voice_notes.csv")
    cache = load_media_cache()

    client = get_anthropic_client()
    for _, row in images.iterrows():
        image_id = row["image_id"]
        if image_id in cache:
            print(f"[skip] {image_id} already cached")
            continue
        print(f"[image] resolving {image_id} ...")
        cache[image_id] = resolve_image(image_id, row["file_path"], client=client)
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
            print(f"{media_id} [{entry['poster_type']}] risk={entry['risk_flags']}")
            print(f"    ocr_text: {entry['ocr_text'][:100]!r}")
        else:
            print(f"{media_id} [voice] transcript: {entry['transcript'][:100]!r}")


if __name__ == "__main__":
    main()

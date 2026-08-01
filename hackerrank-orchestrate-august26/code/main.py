"""Entry point for the Message Notification Router.

Run with: python code/main.py

Pipeline per message (never reads sample_messages.csv, which is for
prompt-writing/evaluation only, not decision logic):
  build_context -> retrieve_evidence -> safety_rules (in priority order) ->
  decision_llm (only if no rule fired) -> calibration -> output row

Writes dataset/output.csv with exactly one row per messages.csv message_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from pipeline.calibration import (
    calibrate_llm_confidence_for_context,
    calibrate_rule_confidence,
    finalize_evidence_ids,
)
from pipeline.context_loader import build_context
from pipeline.data_loader import load_all
from pipeline.decision_llm import make_decision
from pipeline.evidence_retriever import retrieve_evidence
from pipeline.media_resolver import get_gemini_client, load_media_cache
from pipeline.safety_rules import (
    direct_mention_override,
    notification_fatigue,
    quiet_hours_cap,
    scam_phishing_override,
)

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids",
]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "output.csv"

# Checked in this order; the first rule that fires short-circuits the LLM.
SAFETY_RULES = [scam_phishing_override, direct_mention_override, quiet_hours_cap, notification_fatigue]


def _run_safety_rules(ctx, media_cache, bundle):
    for rule in SAFETY_RULES:
        if rule is scam_phishing_override:
            result = rule(ctx, media_cache=media_cache, bundle=bundle)
        else:
            result = rule(ctx, media_cache=media_cache)
        if result is not None:
            return rule.__name__, result
    return None, None


def process_message(row, bundle, media_cache, gemini_client) -> tuple[dict, str]:
    ctx = build_context(row, bundle)
    evidence = retrieve_evidence(ctx)

    rule_name, rule_result = _run_safety_rules(ctx, media_cache, bundle)
    if rule_result is not None:
        action = rule_result.action
        message_type = rule_result.message_type
        reason = rule_result.reason
        confidence = calibrate_rule_confidence(rule_result.confidence)
        source = rule_name
    else:
        decision = make_decision(ctx, media_cache, evidence, gemini_client=gemini_client)
        action = decision.action
        message_type = decision.message_type
        reason = decision.reason
        confidence = calibrate_llm_confidence_for_context(decision.confidence, ctx, evidence)
        source = "llm"

    row_out = {
        "message_id": ctx.message["message_id"],
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": finalize_evidence_ids(evidence),
    }
    return row_out, source


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("Loading dataset and media cache...")
    bundle = load_all()
    media_cache = load_media_cache()
    gemini_client = get_gemini_client()

    messages = bundle.messages  # decision logic only ever reads messages.csv
    total = len(messages)
    print(f"Routing {total} messages...\n")

    output_rows = []
    for i, (_, row) in enumerate(messages.iterrows(), start=1):
        row_out, source = process_message(row, bundle, media_cache, gemini_client)
        output_rows.append(row_out)
        print(f"[{i}/{total}] {row_out['message_id']}: {row_out['action']}/"
              f"{row_out['message_type']} (conf={row_out['confidence']}, via {source})")

    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(output_df)} rows to {OUTPUT_PATH}")

    # Verify the actual persisted file, not just the in-memory frame.
    written = pd.read_csv(OUTPUT_PATH)
    assert len(written) == total, f"Expected {total} rows in output.csv, found {len(written)}"
    assert set(written["message_id"]) == set(messages["message_id"]), (
        "output.csv message_ids don't match messages.csv"
    )
    assert list(written.columns) == OUTPUT_COLUMNS, f"Unexpected columns: {list(written.columns)}"
    print(f"SUCCESS: output.csv has exactly {total} rows, one per messages.csv message_id.")

    print("\n=== action distribution ===")
    print(output_df["action"].value_counts().to_string())
    print("\n=== message_type distribution ===")
    print(output_df["message_type"].value_counts().to_string())


if __name__ == "__main__":
    main()

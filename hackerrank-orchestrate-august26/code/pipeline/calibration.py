"""Final confidence calibration and evidence_message_ids validation — the
last step before a decision (from safety_rules.py or decision_llm.py) is
written to output.csv.

This is deliberately pure post-processing with no LLM call: whether a
message has "strong evidence" or is "genuinely ambiguous" is already fully
determined by evidence_retriever.py's own output and relevant_history, so
mapping that to a confidence band is arithmetic, not judgment. Keeping it
LLM-free also means no extra network dependency for a numeric clamp. If a
model call is ever needed here, it should follow the same Gemini-primary /
Groq-fallback pattern as media_resolver.py and decision_llm.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.context_loader import MessageContext

# Rule-based overrides are deterministic (a regex matched, a flag was set) —
# they should sound sure, but never claim total certainty.
RULE_CONFIDENCE_MIN = 0.85
RULE_CONFIDENCE_MAX = 0.95

# LLM decisions: the sample data never goes above 0.91 or below 0.78, so
# treat that as the healthy band and only widen it where the evidence
# genuinely justifies more or less certainty.
LLM_CONFIDENCE_FLOOR = 0.6
LLM_STRONG_EVIDENCE_CAP = 0.9
LLM_AMBIGUOUS_MIN = 0.6
LLM_AMBIGUOUS_MAX = 0.75
LLM_DEFAULT_MAX = 0.9

MAX_EVIDENCE_IDS = 3


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def calibrate_rule_confidence(raw_confidence: float) -> float:
    """Rule-based overrides are deterministic; sound sure but not absolute."""
    return round(_clamp(raw_confidence, RULE_CONFIDENCE_MIN, RULE_CONFIDENCE_MAX), 2)


def calibrate_llm_confidence(
    raw_confidence: float,
    evidence: List[Tuple[str, str]],
    has_any_history: bool,
) -> float:
    """Caps an LLM-reported confidence based on how much real behavioral
    evidence backs the decision:
    - evidence_retriever found genuinely relevant history -> up to ~0.9
    - no relevant evidence AND no history at all (cold/new sender) -> capped
      to 0.6-0.75, since there's nothing concrete to be confident about
    - no relevant evidence but some history exists -> the general 0.6-0.9
      band, since it's neither strongly backed nor a total cold start
    """
    if evidence:
        return round(_clamp(raw_confidence, LLM_CONFIDENCE_FLOOR, LLM_STRONG_EVIDENCE_CAP), 2)
    if not has_any_history:
        return round(_clamp(raw_confidence, LLM_AMBIGUOUS_MIN, LLM_AMBIGUOUS_MAX), 2)
    return round(_clamp(raw_confidence, LLM_CONFIDENCE_FLOOR, LLM_DEFAULT_MAX), 2)


def calibrate_llm_confidence_for_context(
    raw_confidence: float, ctx: MessageContext, evidence: List[Tuple[str, str]]
) -> float:
    """Convenience wrapper deriving has_any_history from the context."""
    has_any_history = not ctx.relevant_history.empty
    return calibrate_llm_confidence(raw_confidence, evidence, has_any_history)


def finalize_evidence_ids(
    retrieved_evidence: List[Tuple[str, str]],
    candidate_ids: Optional[List[str]] = None,
) -> str:
    """Builds the final semicolon-separated evidence_message_ids field
    (capped at 3), or the literal string 'none'.

    If candidate_ids is given (e.g. a future extension where the LLM picks
    among retrieved evidence rather than us always including all of it),
    every id is validated against retrieved_evidence first — an id that
    evidence_retriever never surfaced for this message is dropped rather
    than trusted, so the LLM can select but never invent an evidence id.
    """
    retrieved_ids = [message_id for message_id, _ in retrieved_evidence]

    if candidate_ids is None:
        selected = retrieved_ids
    else:
        retrieved_set = set(retrieved_ids)
        selected = [i for i in candidate_ids if i in retrieved_set]
        invalid = [i for i in candidate_ids if i not in retrieved_set]
        if invalid:
            print(f"  WARNING: dropped invented evidence id(s) not returned by "
                  f"evidence_retriever: {invalid}")

    selected = selected[:MAX_EVIDENCE_IDS]
    return ";".join(selected) if selected else "none"


if __name__ == "__main__":
    print("=== Rule confidence calibration ===")
    for raw in [0.99, 0.5, 0.9, 0.0]:
        print(f"  raw={raw} -> calibrated={calibrate_rule_confidence(raw)}")

    print("\n=== LLM confidence calibration ===")
    strong_evidence = [("message_0129", "User replied to a similar message ... on the same topic.")]
    print(f"  strong evidence, raw=0.99 -> {calibrate_llm_confidence(0.99, strong_evidence, True)}")
    print(f"  strong evidence, raw=0.4  -> {calibrate_llm_confidence(0.4, strong_evidence, True)}")
    print(f"  no evidence, no history (cold sender), raw=0.95 -> "
          f"{calibrate_llm_confidence(0.95, [], False)}")
    print(f"  no evidence, no history (cold sender), raw=0.3  -> "
          f"{calibrate_llm_confidence(0.3, [], False)}")
    print(f"  no evidence, but some history exists, raw=0.99 -> "
          f"{calibrate_llm_confidence(0.99, [], True)}")

    print("\n=== evidence_message_ids finalization ===")
    many = [("message_0271", "r1"), ("message_0030", "r2"), ("message_0129", "r3"),
            ("message_0189", "r4")]
    print(f"  4 retrieved -> {finalize_evidence_ids(many)!r} (capped at 3)")
    print(f"  none retrieved -> {finalize_evidence_ids([])!r}")

    print("\n=== Rejecting an invented id ===")
    result = finalize_evidence_ids(
        retrieved_evidence=[("message_0129", "r1")],
        candidate_ids=["message_0129", "message_9999"],
    )
    print(f"  candidate included a hallucinated id -> {result!r}")

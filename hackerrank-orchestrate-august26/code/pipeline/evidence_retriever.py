"""Ranks a message's relevant_history (already scoped to the same sender,
group, or business by context_loader) as evidence for the routing decision.

Ranking combines three signals: whether the user actually reacted
(opened/replied/dismissed/muted/reported carries more evidentiary weight than
no recorded signal), recency, and topical text overlap with the current
message — a same-subject message from weeks ago can matter more than an
unrelated one from yesterday."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.context_loader import MessageContext, build_context
from pipeline.data_loader import load_all

MAX_EVIDENCE = 3
RECENCY_HALF_LIFE_DAYS = 30.0
NO_SIGNAL_INCLUSION_DAYS = 30
TEXT_SIMILARITY_WEIGHT = 20.0
TEXT_SIMILARITY_INCLUDE_THRESHOLD = 0.12

_WORD_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "for", "and", "or", "but", "if", "this", "that", "these", "those", "it",
    "its", "be", "will", "with", "from", "your", "you", "please", "pls",
    "thanks", "thank", "now", "just", "so", "we", "he", "she", "they", "them",
    "his", "her", "not", "no", "yes", "can", "could", "would", "should",
    "have", "has", "had", "do", "does", "did", "as", "by", "up", "out",
    "about", "here", "there", "today", "tomorrow", "after", "before", "once",
    "still", "also", "our", "all", "any", "one",
}


def _tokenize(text: Optional[str]) -> set:
    if not text or not isinstance(text, str):
        return set()
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _text_similarity(tokens_a: set, tokens_b: set) -> float:
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)

# Checked in priority order (highest weight first). The first matching
# column with value 1 determines a row's signal weight and reason template.
_SIGNAL_RULES = [
    ("message_reported", 5,
     "User previously reported a similar message from {who} {days}."),
    ("muted_after_message", 4,
     "User muted {who} right after a similar message {days}."),
    ("message_replied", 3,
     "User replied to a similar message from {who} {days}."),
    ("notification_dismissed", 2,
     "User dismissed a similar message from {who} {days} without opening it."),
    ("message_opened", 1,
     "User opened a similar message from {who} {days}."),
]
_NO_SIGNAL_TEMPLATE = "A similar message was sent by {who} {days} (no recorded reaction)."


def _who(ctx: MessageContext) -> str:
    if ctx.group is not None:
        return f"this group ({ctx.group['group_name']})"
    if ctx.business is not None:
        return f"this business ({ctx.business['display_name']})"
    return f"this sender ({ctx.message.get('sender_user_id')})"


def _days_ago_phrase(days: int) -> str:
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    return f"{days} days ago"


def _score_row(row: pd.Series, current_time) -> Tuple[int, float, str, int]:
    """Returns (signal_weight, recency_score, reason_template, days_ago)."""
    days = max((current_time - row["created_at"]).days, 0)
    recency_score = math.exp(-days / RECENCY_HALF_LIFE_DAYS)

    for column, weight, template in _SIGNAL_RULES:
        value = row.get(column)
        if pd.notna(value) and value == 1:
            return weight, recency_score, template, days

    return 0, recency_score, _NO_SIGNAL_TEMPLATE, days


def retrieve_evidence(
    ctx: MessageContext, max_results: int = MAX_EVIDENCE
) -> List[Tuple[str, str]]:
    """Returns up to max_results (message_id, reason) pairs ranked most-
    relevant first, or [] if nothing in the history is genuinely relevant."""
    history = ctx.relevant_history
    if history.empty:
        return []

    current_time = ctx.message["created_at"]
    current_tokens = _tokenize(ctx.message.get("message_text"))
    who = _who(ctx)

    candidates = []
    for _, row in history.iterrows():
        weight, recency_score, template, days = _score_row(row, current_time)
        similarity = _text_similarity(current_tokens, _tokenize(row.get("message_text")))
        is_topical_match = similarity >= TEXT_SIMILARITY_INCLUDE_THRESHOLD

        if weight == 0 and days > NO_SIGNAL_INCLUSION_DAYS and not is_topical_match:
            continue  # no reaction, too stale, and not even on the same topic

        score = weight * 10 + recency_score + similarity * TEXT_SIMILARITY_WEIGHT
        reason = template.format(who=who, days=_days_ago_phrase(days))
        if is_topical_match:
            reason = reason.rstrip(".") + " on the same topic."
        candidates.append((score, row["message_id"], reason))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [(message_id, reason) for _, message_id, reason in candidates[:max_results]]


if __name__ == "__main__":
    bundle = load_all()
    for message_id in ["sample_msg_001", "sample_msg_004", "sample_msg_049"]:
        row = bundle.sample_messages[bundle.sample_messages["message_id"] == message_id].iloc[0]
        ctx = build_context(row, bundle)
        evidence = retrieve_evidence(ctx)
        print(f"=== {message_id} ({row['conversation_type']}) — "
              f"{len(ctx.relevant_history)} candidates in history ===")
        if not evidence:
            print("  none")
        for evidence_id, reason in evidence:
            print(f"  {evidence_id}: {reason}")
        print()

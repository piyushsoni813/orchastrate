"""Hard-coded safety/policy rules that run before any LLM call. Each rule is
independent and self-contained: given a MessageContext (and, for media
messages, the resolved media_cache), it either fires with a full decision or
returns None because it doesn't apply. Callers run them in priority order and
stop at the first fire, short-circuiting the LLM entirely.

Rules deliberately never consult engagement-history fields (messages_opened,
messages_replied, etc.) to *soften* a fired decision — see scam_phishing_override.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.context_loader import MessageContext, build_context
from pipeline.data_loader import DataBundle, load_all
from pipeline.media_resolver import load_media_cache


class RuleResult(NamedTuple):
    fires: bool
    action: str
    message_type: str
    reason: str
    confidence: float


OTP_PATTERN = re.compile(r"\botp\b|one[- ]?time password|verification code", re.I)
URGENCY_PATTERNS = re.compile(
    r"\b(urgent|immediately|right away|act now|expire[sd]?\s*(today|soon|shortly)?|"
    r"blocked?\s*(today|now|unless)|verify\s*(now|immediately)|login\s*now|"
    r"last chance|final (notice|warning)|suspend(ed)?\s*(today|immediately)?|"
    r"within \d+\s*(minutes?|hours?|mins?))\b",
    re.I,
)
SUSPICIOUS_DOMAIN_AGE_DAYS = 30

FATIGUE_MULTIPLIER = 1.5
MIN_NOTIFICATIONS_SENT_TO_FLAG = 5


def _effective_text(ctx: MessageContext, media_cache: Optional[dict]) -> str:
    """Message text plus OCR'd/transcribed media text, so a scam or urgency
    signal hiding in an image poster or voice note isn't missed."""
    parts = [ctx.message.get("message_text") or ""]
    media_id = ctx.message.get("media_id")
    if media_cache and media_id:
        media = media_cache.get(media_id)
        if media:
            if media.get("media_type") == "image":
                parts.append(media.get("ocr_text") or "")
            elif media.get("media_type") == "voice":
                parts.append(media.get("transcript") or "")
    return " ".join(p for p in parts if p)


def _image_risk_flags(ctx: MessageContext, media_cache: Optional[dict]) -> List[str]:
    media_id = ctx.message.get("media_id")
    if not media_cache or not media_id:
        return []
    media = media_cache.get(media_id)
    if not media or media.get("media_type") != "image":
        return []
    return media.get("risk_flags") or []


def _has_urgency_or_risk_signal(text: str, image_risk_flags: List[str]) -> bool:
    # OTP mentions are treated as an inherent risk signal on their own (not
    # just when phrased with English urgency words) — msg_070 in the sample
    # data is Hindi/English mixed ("OTP leak ho gaya hai... account block ho
    # jayega") and only the literal "OTP" token is reliably catchable.
    return (
        bool(URGENCY_PATTERNS.search(text))
        or bool(OTP_PATTERN.search(text))
        or bool(image_risk_flags)
    )


def scam_phishing_override(
    ctx: MessageContext,
    media_cache: Optional[dict] = None,
    bundle: Optional[DataBundle] = None,
) -> Optional[RuleResult]:
    """OTP request + urgency language + an identity/trust red flag ->
    hard mute as scam. This never looks at how much the user usually
    engages with the sender — engagement history is explicitly NOT one of
    the conditions below, by design, per the problem statement."""
    text = _effective_text(ctx, media_cache)
    image_risk_flags = _image_risk_flags(ctx, media_cache)

    has_otp_request = bool(OTP_PATTERN.search(text))
    has_urgency = bool(URGENCY_PATTERNS.search(text)) or "urgency_language" in image_risk_flags
    if not (has_otp_request and has_urgency):
        return None

    identity_risk_reasons: List[str] = []
    business = ctx.business
    conversation_type = ctx.message.get("conversation_type")

    if business is not None:
        if not business.get("verified"):
            identity_risk_reasons.append("the business is not verified")
        if business.get("domain_used_by_sender") != business.get("official_domain"):
            identity_risk_reasons.append(
                "the sender's domain does not match the business's official domain"
            )
        domain_age = business.get("domain_used_by_sender_age_days")
        if domain_age is not None and domain_age < SUSPICIOUS_DOMAIN_AGE_DAYS:
            identity_risk_reasons.append(f"the sender's domain is only {domain_age} days old")
        if ctx.user_business_history is None:
            identity_risk_reasons.append("the user has no prior relationship with this business")
    elif conversation_type == "group":
        sender_id = ctx.message.get("sender_user_id")
        group_id = ctx.message.get("group_id")
        if bundle is not None and sender_id and group_id:
            if bundle.get_group_membership(group_id, sender_id) is None:
                identity_risk_reasons.append("the sender is not a known member of this group")
    else:  # personal
        if ctx.relevant_history.empty:
            identity_risk_reasons.append("the user has no prior message history with this sender")

    if not identity_risk_reasons:
        return None

    reason = (
        "OTP request combined with urgency language from an untrustworthy sender: "
        + "; ".join(identity_risk_reasons)
        + "."
    )
    return RuleResult(
        fires=True, action="mute", message_type="scam", reason=reason, confidence=0.95
    )


def direct_mention_override(
    ctx: MessageContext, media_cache: Optional[dict] = None
) -> Optional[RuleResult]:
    """A muted group is a blanket suppressor UNLESS this message directly
    @mentions the receiving user — check that before treating the mute as
    final. Mentions in this dataset appear as a literal "@{user_id}" token."""
    if ctx.message.get("conversation_type") != "group":
        return None
    if ctx.group_membership is None or not ctx.group_membership.get("group_muted_by_user"):
        return None

    text = _effective_text(ctx, media_cache)
    user_id = str(ctx.message.get("user_id"))
    mention_pattern = re.compile(rf"@{re.escape(user_id)}\b", re.I)
    if not mention_pattern.search(text):
        return None

    return RuleResult(
        fires=True,
        action="notify",
        message_type="personal",
        reason=f"Group is muted, but the message directly @mentions {user_id}, "
        f"so it should still interrupt.",
        confidence=0.8,
    )


def quiet_hours_cap(
    ctx: MessageContext, media_cache: Optional[dict] = None
) -> Optional[RuleResult]:
    """During the user's do-not-disturb window, cap at 'digest' unless the
    message itself shows urgency or risk (a real scam/urgent message during
    quiet hours should still be free to escalate past this cap)."""
    if ctx.user is None:
        return None
    created_at = ctx.message.get("created_at")
    if created_at is None or not ctx.is_within_dnd(created_at):
        return None

    text = _effective_text(ctx, media_cache)
    image_risk_flags = _image_risk_flags(ctx, media_cache)
    if _has_urgency_or_risk_signal(text, image_risk_flags):
        return None

    return RuleResult(
        fires=True,
        action="digest",
        message_type="unknown",
        reason="Message arrived during the user's quiet hours and shows no "
        "urgency or risk signal, so it's capped at digest instead of notify.",
        confidence=0.7,
    )


def notification_fatigue(
    ctx: MessageContext, media_cache: Optional[dict] = None
) -> Optional[RuleResult]:
    """If the user's most recent known day of notifications is well above
    their recent daily average and this message is low-stakes, prefer
    digest over notify. daily_notification_summary.csv covers a fixed
    ~14-day window per user rather than being keyed to each message's own
    date, so "today" here means the most recent day on record for the
    user, not necessarily the calendar date of this specific message."""
    summary = ctx.daily_notification_summary  # sorted most-recent-first
    if len(summary) < 2:
        return None

    notifications_latest = summary.iloc[0]["notifications_sent"]
    recent_average = summary.iloc[1:]["notifications_sent"].mean()

    is_heavily_notified = (
        notifications_latest >= MIN_NOTIFICATIONS_SENT_TO_FLAG
        and recent_average > 0
        and notifications_latest > recent_average * FATIGUE_MULTIPLIER
    )
    if not is_heavily_notified:
        return None

    text = _effective_text(ctx, media_cache)
    image_risk_flags = _image_risk_flags(ctx, media_cache)
    is_low_stakes = (
        ctx.message.get("conversation_type") != "personal"
        and not _has_urgency_or_risk_signal(text, image_risk_flags)
    )
    if not is_low_stakes:
        return None

    return RuleResult(
        fires=True,
        action="digest",
        message_type="unknown",
        reason=f"User already received {notifications_latest} notifications on their "
        f"most recent day (vs a {recent_average:.1f} recent daily average) and this "
        f"message shows no urgency, so it's deferred to a digest.",
        confidence=0.6,
    )


if __name__ == "__main__":
    bundle = load_all()
    media_cache = load_media_cache()
    rules = [scam_phishing_override, direct_mention_override, quiet_hours_cap, notification_fatigue]

    test_message_ids = [
        "msg_085", "msg_070", "msg_056", "msg_040", "msg_101", "msg_016", "msg_063", "msg_077",
    ]
    for message_id in test_message_ids:
        row = bundle.messages[bundle.messages["message_id"] == message_id].iloc[0]
        ctx = build_context(row, bundle)
        print(f"=== {message_id} ({row['conversation_type']}) "
              f"text={str(row['message_text'])[:60]!r} media={row['media_type']} ===")
        fired_any = False
        for rule in rules:
            if rule is scam_phishing_override:
                result = rule(ctx, media_cache=media_cache, bundle=bundle)
            else:
                result = rule(ctx, media_cache=media_cache)
            if result is not None:
                fired_any = True
                print(f"  [{rule.__name__}] -> {result.action}/{result.message_type} "
                      f"(conf={result.confidence}): {result.reason}")
        if not fired_any:
            print("  no safety rule fired")
        print()

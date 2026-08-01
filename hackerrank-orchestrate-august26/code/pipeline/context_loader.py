"""Builds a MessageContext for a single message row (from messages.csv or
sample_messages.csv — both share the same input columns) by joining it
against every other dataset already indexed in a DataBundle."""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_loader import DataBundle, is_within_dnd as _is_within_dnd_window, load_all

_EVENT_COLUMNS = [
    "user_id",
    "message_id",
    "message_opened",
    "message_replied",
    "reaction_time_minutes",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
]


def _clean(value):
    """Recursively replaces pandas NaN/NaT with None so missing optional
    fields (e.g. no user_business_history row) are unambiguous."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _make_dnd_checker(user: Optional[dict]) -> Callable[[dt.datetime], bool]:
    if user is None:
        def _missing(_timestamp: dt.datetime) -> bool:
            raise ValueError("Cannot evaluate DND window: user not found")

        return _missing

    start = user["dnd_start"]
    end = user["dnd_end"]

    def _checker(timestamp: dt.datetime) -> bool:
        check_time = timestamp.time() if isinstance(timestamp, (dt.datetime, pd.Timestamp)) else timestamp
        return _is_within_dnd_window(check_time, start, end)

    return _checker


@dataclass
class MessageContext:
    message: dict
    user: Optional[dict]
    is_within_dnd: Callable[[dt.datetime], bool]
    group: Optional[dict] = None
    group_membership: Optional[dict] = None
    business: Optional[dict] = None
    user_business_history: Optional[dict] = None
    relevant_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_notification_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_relevant_history(message: dict, bundle: DataBundle) -> pd.DataFrame:
    """This user's message_history rows scoped to whichever counterpart
    applies to this message's conversation_type, joined with how the user
    reacted to each (message_events)."""
    user_id = message["user_id"]
    conversation_type = message.get("conversation_type")
    history = bundle.get_history_for_user(user_id)

    if conversation_type == "personal":
        history = history[history["sender_user_id"] == message.get("sender_user_id")]
    elif conversation_type == "group":
        history = history[history["group_id"] == message.get("group_id")]
    elif conversation_type == "business":
        history = history[history["business_id"] == message.get("business_id")]
    else:
        history = history.iloc[0:0]

    if history.empty:
        return history

    events = bundle.message_events[_EVENT_COLUMNS]
    merged = history.merge(events, on=["user_id", "message_id"], how="left")
    return merged.sort_values("created_at", ascending=False).reset_index(drop=True)


def build_context(row: pd.Series, bundle: DataBundle) -> MessageContext:
    message = _clean(row.to_dict())
    user_id = message["user_id"]
    user = _clean(bundle.get_user(user_id))
    conversation_type = message.get("conversation_type")

    group = None
    group_membership = None
    business = None
    user_business_history = None

    if conversation_type == "group" and message.get("group_id"):
        group = _clean(bundle.get_group(message["group_id"]))
        group_membership = _clean(bundle.get_group_membership(message["group_id"], user_id))
    elif conversation_type == "business" and message.get("business_id"):
        business = _clean(bundle.get_business(message["business_id"]))
        user_business_history = _clean(
            bundle.get_user_business_history(user_id, message["business_id"])
        )

    daily_summary = bundle.get_daily_summary_for_user(user_id).sort_values(
        "date", ascending=False
    )

    return MessageContext(
        message=message,
        user=user,
        is_within_dnd=_make_dnd_checker(user),
        group=group,
        group_membership=group_membership,
        business=business,
        user_business_history=user_business_history,
        relevant_history=_build_relevant_history(message, bundle),
        daily_notification_summary=daily_summary,
    )


def _print_context(ctx: MessageContext) -> None:
    msg = ctx.message
    print(f"message_id={msg['message_id']}  conversation_type={msg['conversation_type']}  "
          f"created_at={msg['created_at']}")
    print(f"  text: {str(msg.get('message_text'))[:80]!r}")

    if ctx.user is None:
        print("  user: NOT FOUND")
    else:
        print(f"  user: {msg['user_id']}  dnd_window={ctx.user['do_not_disturb_window']}  "
              f"is_within_dnd(created_at)={ctx.is_within_dnd(msg['created_at'])}")

    if ctx.group is not None:
        print(f"  group: {ctx.group['group_name']} ({ctx.group['group_type']}, "
              f"{ctx.group['member_count']} members)")
        print(f"  group_membership: {ctx.group_membership}")

    if ctx.business is not None:
        print(f"  business: {ctx.business['display_name']} "
              f"(verified={ctx.business['verified']})")
        print(f"  user_business_history: {ctx.user_business_history}")

    print(f"  relevant_history: {len(ctx.relevant_history)} rows")
    if not ctx.relevant_history.empty:
        cols = ["message_id", "created_at", "message_text", "message_opened",
                "message_replied", "message_reported"]
        print(ctx.relevant_history[cols].head(5).to_string(index=False))

    print(f"  daily_notification_summary: {len(ctx.daily_notification_summary)} rows")
    if not ctx.daily_notification_summary.empty:
        print(ctx.daily_notification_summary[["date", "notifications_sent",
                                                "notifications_dismissed"]]
              .head(3).to_string(index=False))
    print()


if __name__ == "__main__":
    bundle = load_all()
    for conv_type in ["group", "business", "personal"]:
        sample = bundle.sample_messages[bundle.sample_messages["conversation_type"] == conv_type]
        if sample.empty:
            print(f"No sample_messages row found for conversation_type={conv_type}")
            continue
        row = sample.iloc[0]
        print(f"=== {conv_type} sample: {row['message_id']} ===")
        ctx = build_context(row, bundle)
        _print_context(ctx)

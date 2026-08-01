"""Loads and indexes every dataset CSV once so later pipeline stages do O(1)
dict/DataFrame lookups instead of re-reading files or re-scanning rows."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"

_DATETIME_COLUMNS: Dict[str, list] = {
    "messages.csv": ["created_at"],
    "sample_messages.csv": ["created_at"],
    "message_history.csv": ["created_at"],
    "groups.csv": ["created_at"],
    "group_members.csv": ["joined_at"],
    "user_business_history.csv": [
        "last_activity_at",
        "promotions_opted_out_at",
        "last_reply_at",
    ],
    "daily_notification_summary.csv": ["date"],
}


def _read_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / filename)
    for col in _DATETIME_COLUMNS.get(filename, []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def parse_dnd_window(window: str) -> Tuple[dt.time, dt.time]:
    """Parses "HH:MM-HH:MM" into (start, end) times. The window may cross
    midnight (e.g. "22:00-07:00"), which callers must handle when checking
    membership rather than assuming start < end."""
    start_str, end_str = window.split("-")
    start = dt.datetime.strptime(start_str.strip(), "%H:%M").time()
    end = dt.datetime.strptime(end_str.strip(), "%H:%M").time()
    return start, end


def is_within_dnd(check_time: dt.time, start: dt.time, end: dt.time) -> bool:
    """True if check_time falls inside [start, end), handling windows that
    cross midnight (start > end)."""
    if start <= end:
        return start <= check_time < end
    return check_time >= start or check_time < end


def _add_dnd_columns(users: pd.DataFrame) -> pd.DataFrame:
    parsed = users["do_not_disturb_window"].apply(parse_dnd_window)
    users["dnd_start"] = parsed.apply(lambda p: p[0])
    users["dnd_end"] = parsed.apply(lambda p: p[1])
    return users


@dataclass
class DataBundle:
    messages: pd.DataFrame
    sample_messages: pd.DataFrame
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame

    # Point lookups: key -> dict of column -> value.
    users_by_id: Dict[str, dict]
    groups_by_id: Dict[str, dict]
    business_accounts_by_id: Dict[str, dict]
    images_by_id: Dict[str, dict]
    voice_notes_by_id: Dict[str, dict]
    group_membership: Dict[Tuple[str, str], dict]
    user_business_by_pair: Dict[Tuple[str, str], dict]
    message_events_by_pair: Dict[Tuple[str, str], dict]

    # Grouped lookups: key -> DataFrame of all matching rows.
    group_members_by_group: Dict[str, pd.DataFrame]
    message_history_by_user: Dict[str, pd.DataFrame]
    user_business_history_by_user: Dict[str, pd.DataFrame]
    daily_notification_summary_by_user: Dict[str, pd.DataFrame]

    def get_user(self, user_id: str) -> Optional[dict]:
        return self.users_by_id.get(user_id)

    def get_group(self, group_id: str) -> Optional[dict]:
        return self.groups_by_id.get(group_id)

    def get_business(self, business_id: str) -> Optional[dict]:
        return self.business_accounts_by_id.get(business_id)

    def get_group_membership(self, group_id: str, user_id: str) -> Optional[dict]:
        return self.group_membership.get((group_id, user_id))

    def get_user_business_history(self, user_id: str, business_id: str) -> Optional[dict]:
        return self.user_business_by_pair.get((user_id, business_id))

    def get_message_event(self, user_id: str, message_id: str) -> Optional[dict]:
        return self.message_events_by_pair.get((user_id, message_id))

    def get_history_for_user(self, user_id: str) -> pd.DataFrame:
        return self.message_history_by_user.get(user_id, self.message_history.iloc[0:0])

    def get_daily_summary_for_user(self, user_id: str) -> pd.DataFrame:
        return self.daily_notification_summary_by_user.get(
            user_id, self.daily_notification_summary.iloc[0:0]
        )


def load_all() -> DataBundle:
    messages = _read_csv("messages.csv")
    sample_messages = _read_csv("sample_messages.csv")
    users = _add_dnd_columns(_read_csv("users.csv"))
    groups = _read_csv("groups.csv")
    group_members = _read_csv("group_members.csv")
    business_accounts = _read_csv("business_accounts.csv")
    user_business_history = _read_csv("user_business_history.csv")
    message_history = _read_csv("message_history.csv")
    message_events = _read_csv("message_events.csv")
    images = _read_csv("images.csv")
    voice_notes = _read_csv("voice_notes.csv")
    daily_notification_summary = _read_csv("daily_notification_summary.csv")

    return DataBundle(
        messages=messages,
        sample_messages=sample_messages,
        users=users,
        groups=groups,
        group_members=group_members,
        business_accounts=business_accounts,
        user_business_history=user_business_history,
        message_history=message_history,
        message_events=message_events,
        images=images,
        voice_notes=voice_notes,
        daily_notification_summary=daily_notification_summary,
        users_by_id=users.set_index("user_id").to_dict("index"),
        groups_by_id=groups.set_index("group_id").to_dict("index"),
        business_accounts_by_id=business_accounts.set_index("business_id").to_dict("index"),
        images_by_id=images.set_index("image_id").to_dict("index"),
        voice_notes_by_id=voice_notes.set_index("voice_note_id").to_dict("index"),
        group_membership=group_members.set_index(["group_id", "user_id"]).to_dict("index"),
        user_business_by_pair=user_business_history.set_index(
            ["user_id", "business_id"]
        ).to_dict("index"),
        message_events_by_pair=message_events.set_index(["user_id", "message_id"]).to_dict(
            "index"
        ),
        group_members_by_group={
            key: group for key, group in group_members.groupby("group_id")
        },
        message_history_by_user={
            key: group for key, group in message_history.groupby("user_id")
        },
        user_business_history_by_user={
            key: group for key, group in user_business_history.groupby("user_id")
        },
        daily_notification_summary_by_user={
            key: group for key, group in daily_notification_summary.groupby("user_id")
        },
    )


if __name__ == "__main__":
    bundle = load_all()
    frames = {
        "messages": bundle.messages,
        "sample_messages": bundle.sample_messages,
        "users": bundle.users,
        "groups": bundle.groups,
        "group_members": bundle.group_members,
        "business_accounts": bundle.business_accounts,
        "user_business_history": bundle.user_business_history,
        "message_history": bundle.message_history,
        "message_events": bundle.message_events,
        "images": bundle.images,
        "voice_notes": bundle.voice_notes,
        "daily_notification_summary": bundle.daily_notification_summary,
    }
    for name, df in frames.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        print(df.dtypes)

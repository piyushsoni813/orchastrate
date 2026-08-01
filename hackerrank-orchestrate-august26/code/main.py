"""Entry point for the Message Notification Router.

Run with: python code/main.py
"""

from __future__ import annotations

from pipeline.data_loader import load_all


def main() -> None:
    bundle = load_all()
    print(f"Loaded {len(bundle.messages)} messages to route.")
    print(f"Loaded {len(bundle.users)} users, {len(bundle.groups)} groups, "
          f"{len(bundle.business_accounts)} business accounts.")
    # TODO: run routing pipeline over bundle.messages and write dataset/output.csv


if __name__ == "__main__":
    main()

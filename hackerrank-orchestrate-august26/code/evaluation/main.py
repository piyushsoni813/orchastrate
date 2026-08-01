"""Entry point for evaluating the router against dataset/sample_messages.csv.

Run with: python code/evaluation/main.py

sample_messages.csv uses its own sample_msg_XXX ids, which never overlap with
messages.csv ids, so evaluation must run the pipeline on these rows directly
and compare against their given labels rather than joining by id.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_loader import load_all


def main() -> None:
    bundle = load_all()
    print(f"Loaded {len(bundle.sample_messages)} labeled sample messages for evaluation.")
    # TODO: run the routing pipeline on bundle.sample_messages and score
    # predictions against their action/message_type/reason/confidence/
    # evidence_message_ids columns.


if __name__ == "__main__":
    main()

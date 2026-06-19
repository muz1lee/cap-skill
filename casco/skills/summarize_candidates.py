"""CLI for summarizing CaSCo candidate JSONL inventories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from casco.skills.inventory import read_candidate_records, summarize_candidate_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_candidate_records(read_candidate_records(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote candidate summary to {args.output}")


if __name__ == "__main__":
    main()

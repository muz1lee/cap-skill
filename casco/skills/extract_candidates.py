"""CLI for extracting CaSCo candidate records from one CaP-X run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from casco.skills.candidates import candidate_records_from_run, write_candidate_records_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = candidate_records_from_run(args.run_dir)
    count = write_candidate_records_jsonl(args.output, records)
    print(f"wrote {count} candidate records to {args.output}")


if __name__ == "__main__":
    main()

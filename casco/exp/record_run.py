"""CLI for appending a CaP-X run directory to a CaSCo ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from casco.exp.ledger import append_entry
from casco.exp.runs import build_ledger_entry_from_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--note")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--logged-at")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entry = build_ledger_entry_from_run(
        args.run_dir,
        phase=args.phase,
        status=args.status,
        stage=args.stage,
        note=args.note,
        exit_code=args.exit_code,
    )
    append_entry(
        args.ledger,
        entry,
        now=(lambda: args.logged_at) if args.logged_at is not None else None,
    )
    print(f"appended run {entry['run_id']} to {args.ledger}")


if __name__ == "__main__":
    main()

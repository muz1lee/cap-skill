"""CLI for generating one CaSCo LIBERO config from a working template."""

from __future__ import annotations

import argparse
from pathlib import Path

from casco.bench.libero_configs import materialize_privileged_minimal_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    materialize_privileged_minimal_config(
        template_path=args.template,
        output_path=args.output,
        suite_name=args.suite_name,
        task_id=args.task_id,
        output_dir=args.output_dir,
    )
    print(f"wrote LIBERO config to {args.output}")


if __name__ == "__main__":
    main()

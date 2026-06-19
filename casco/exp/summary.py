"""Parsers for CaP-X run summary files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_summary_file(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    metrics: dict[str, Any] = {}
    for index, line in enumerate(lines):
        if line.startswith("Model:"):
            metrics["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("Config Path:"):
            metrics["config_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("Git Commit:"):
            match = re.match(r"Git Commit:\s+(\S+)\s+\(Dirty:\s+(True|False)\)", line)
            if match:
                metrics["git_commit"] = match.group(1)
                metrics["git_dirty"] = match.group(2) == "True"
        elif line.startswith("Total number of trials:"):
            metrics["total_trials"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Code generation success rate / Average reward / Task completed:"):
            values = lines[index + 1].split("/")
            metrics["code_generation_success_rate"] = float(values[0])
            metrics["average_reward"] = float(values[1])
            metrics["task_completed_count"] = int(values[2])
        elif line.startswith("Average code blocks:"):
            metrics["average_code_blocks"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Average regenerations:"):
            metrics["average_regenerations"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Average finishes:"):
            metrics["average_finishes"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Elapsed time:"):
            metrics["elapsed_time_seconds"] = float(
                line.removeprefix("Elapsed time:").removesuffix("seconds").strip()
            )

    return metrics

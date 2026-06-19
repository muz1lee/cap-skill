"""Materialize CaSCo LIBERO config variants from the working Batch1 template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def materialize_privileged_minimal_config(
    *,
    template_path: str | Path,
    output_path: str | Path,
    suite_name: str,
    task_id: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a privileged minimal LIBERO config for one suite/task pair."""
    template = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    low_level = template["env"]["cfg"]["low_level"]
    low_level["suite_name"] = suite_name
    low_level["task_id"] = task_id
    template["output_dir"] = str(output_dir)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return template

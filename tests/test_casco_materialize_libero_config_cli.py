from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_materialize_libero_config_cli_writes_yaml(tmp_path: Path):
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "env": {"cfg": {"low_level": {"suite_name": "libero_spatial", "task_id": 0}}},
                "output_dir": "./outputs/old",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generated" / "task_02.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.bench.materialize_libero_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--suite-name",
            "libero_goal",
            "--task-id",
            "2",
            "--output-dir",
            "/mnt/nas/wenqian/cap-skill-data/outputs/run_002",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["env"]["cfg"]["low_level"]["suite_name"] == "libero_goal"
    assert generated["env"]["cfg"]["low_level"]["task_id"] == 2
    assert generated["output_dir"] == "/mnt/nas/wenqian/cap-skill-data/outputs/run_002"
    assert str(output) in result.stdout

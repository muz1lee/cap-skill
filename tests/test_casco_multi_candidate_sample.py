from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from casco.exp.sampling.sample_plan import SamplingPlanConfig, build_sampling_plan


def test_build_sampling_plan_creates_one_run_per_task_sample_without_secrets(tmp_path: Path):
    key_file = tmp_path / ".geminikey"
    key_file.write_text("SECRET_KEY\n", encoding="utf-8")
    config = SamplingPlanConfig(
        suite_name="libero_spatial",
        task_ids=(3, 4),
        samples_per_task=2,
        stage="batch2_5_multisample",
        run_group_id="20260619_120000",
        data_root=tmp_path / "data",
        template_path=Path("env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml"),
        key_file=key_file,
        cuda_visible_devices="2",
        uv_bin="/root/.local/bin/uv",
    )

    plan = build_sampling_plan(config)

    assert [spec.task_id for spec in plan] == [3, 3, 4, 4]
    assert [spec.sample_index for spec in plan] == [0, 1, 0, 1]
    assert plan[0].run_id == "batch2_5_multisample_libero_spatial_task3_sample00_20260619_120000"
    assert plan[0].config_path == tmp_path / "data" / "configs" / f"{plan[0].run_id}.yaml"
    assert plan[0].candidate_jsonl == tmp_path / "data" / "candidates" / f"{plan[0].run_id}.jsonl"
    assert plan[0].expected_output_dir == (
        tmp_path / "data" / "outputs" / "gemini-3.5-flash" / plan[0].run_id
    )
    assert "SECRET_KEY" not in json.dumps([spec.redacted_dict() for spec in plan])
    assert plan[0].redacted_dict()["env_overrides"]["OPENAI_API_KEY"] == "<redacted>"


def test_multi_candidate_sample_cli_dry_run_writes_plan(tmp_path: Path):
    key_file = tmp_path / ".geminikey"
    key_file.write_text("SECRET_KEY\n", encoding="utf-8")
    plan_output = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.exp.sampling.multi_candidate_sample",
            "--suite",
            "libero_spatial",
            "--task-ids",
            "3",
            "4",
            "--samples-per-task",
            "1",
            "--stage",
            "batch2_5_multisample",
            "--run-group-id",
            "20260619_121500",
            "--data-root",
            str(tmp_path / "data"),
            "--key-file",
            str(key_file),
            "--uv-bin",
            "/root/.local/bin/uv",
            "--dry-run",
            "--plan-output",
            str(plan_output),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert [item["task_id"] for item in plan["runs"]] == [3, 4]
    assert "SECRET_KEY" not in result.stdout
    assert "SECRET_KEY" not in plan_output.read_text(encoding="utf-8")
    assert str(plan_output) in result.stdout

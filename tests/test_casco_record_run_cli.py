from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from casco.exp.ledger import read_entries


def _write_run_artifacts(run_dir: Path) -> Path:
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "code.py").write_text("print('generated')\n", encoding="utf-8")
    (run_dir / "summaries.txt").write_text(
        "\n".join(
            [
                "Model: gemini-3.5-flash",
                "Config Path: env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
                "Git Commit: f3e9e20 (Dirty: False)",
                "Total number of trials: 1",
                "Code generation success rate / Average reward / Task completed: ",
                "1.000/0.000/0",
                "Average code blocks: 1.000",
                "Average regenerations: 0.000",
                "Average finishes: 0.000",
                "Elapsed time: 123.22 seconds",
            ]
        ),
        encoding="utf-8",
    )
    return trial_dir


def test_record_run_cli_appends_entry_to_ledger(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "gemini-3.5-flash" / "casco_batch1_smoke_20260619_210257"
    _write_run_artifacts(run_dir)
    ledger_path = tmp_path / "ledger.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.exp.record_run",
            "--run-dir",
            str(run_dir),
            "--ledger",
            str(ledger_path),
            "--phase",
            "Batch1",
            "--status",
            "KEPT",
            "--stage",
            "end_to_end_trial",
            "--note",
            "Batch1 success.",
            "--exit-code",
            "0",
            "--logged-at",
            "2026-06-19T13:06:52+00:00",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    entries = read_entries(ledger_path)
    assert len(entries) == 1
    assert entries[0]["run_id"] == "casco_batch1_smoke_20260619_210257"
    assert entries[0]["phase"] == "Batch1"
    assert entries[0]["status"] == "KEPT"
    assert entries[0]["result"]["exit_code"] == 0
    assert entries[0]["logged_at"] == "2026-06-19T13:06:52+00:00"
    assert str(ledger_path) in result.stdout

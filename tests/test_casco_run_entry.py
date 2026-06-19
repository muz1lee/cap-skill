from pathlib import Path

from casco.exp.runs import build_ledger_entry_from_run


def test_build_ledger_entry_from_run_uses_summary_and_trial_artifacts(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "gemini-3.5-flash" / "casco_batch1_smoke_20260619_210257"
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "code.py").write_text("print('generated')\n", encoding="utf-8")
    (run_dir / "summaries.txt").write_text(
        "\n".join(
            [
                "Model: gemini-3.5-flash",
                "Config Path: env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
                "Git Commit: f3e9e20 (Dirty: True)",
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

    entry = build_ledger_entry_from_run(
        run_dir,
        phase="Batch1",
        status="KEPT",
        stage="end_to_end_trial",
        note="Batch1 success.",
        exit_code=0,
    )

    assert entry == {
        "run_id": "casco_batch1_smoke_20260619_210257",
        "phase": "Batch1",
        "status": "KEPT",
        "stage": "end_to_end_trial",
        "model": "gemini-3.5-flash",
        "config": "env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
        "commit": "f3e9e20",
        "git_dirty": True,
        "output_dir": str(run_dir),
        "code_path": str(trial_dir / "code.py"),
        "note": "Batch1 success.",
        "result": {
            "exit_code": 0,
            "code_generation_success_rate": 1.0,
            "reward": 0.0,
            "task_completed": False,
            "task_completed_count": 0,
            "total_trials": 1,
            "elapsed_time_seconds": 123.22,
        },
    }

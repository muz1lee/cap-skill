from pathlib import Path

from casco.exp.summary import parse_summary_file


def test_parse_summary_file_extracts_run_metrics(tmp_path: Path):
    summary = tmp_path / "summaries.txt"
    summary.write_text(
        "\n".join(
            [
                "Summary Statistics:",
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

    assert parse_summary_file(summary) == {
        "model": "gemini-3.5-flash",
        "config_path": "env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
        "git_commit": "f3e9e20",
        "git_dirty": True,
        "total_trials": 1,
        "code_generation_success_rate": 1.0,
        "average_reward": 0.0,
        "task_completed_count": 0,
        "average_code_blocks": 1.0,
        "average_regenerations": 0.0,
        "average_finishes": 0.0,
        "elapsed_time_seconds": 123.22,
    }

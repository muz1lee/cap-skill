"""Batch launcher for CaP-Bench reproduction sweeps.

Drives ``capx/envs/launch.py`` over a hard-coded sweep of (scene x tier)
YAMLs against one or more models, with a small process pool that bounds the
total parallel ``--num-workers`` budget.

Example::

    uv run --no-sync --active capx/envs/scripts/batch_run.py \
        --models qwen3.5-35b-a3b google/gemini-3.1-pro-preview \
        --exp-name reproduce_v1 --total-trials 100 \
        --num-workers 12 --max-parallel-launches 2

Tier mapping (paper -> filename suffix)::

    S1 = privileged                (oracle upper bound)
    S4 = reduced_api_exampleless   (single-turn, no in-context examples)
    M2 = multiturn_vf              (multi-turn + visual feedback)
    M3 = multiturn_vdm             (multi-turn + visual differencing)

Use ``--dry-run`` to print the resolved command list before firing anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Sweep definition (edit here, not on the CLI)
# ---------------------------------------------------------------------------

# Tier -> default filename suffix. S4 is overridden per-scene below because
# two_arm_* configs use ``reduced_exampleless`` (no ``_api``) instead of
# ``reduced_api_exampleless``.
TIER_SUFFIX = {
    "S1": "privileged",
    "S4": "reduced_api_exampleless",
    "M2": "multiturn_vf",
    "M3": "multiturn_vdm",
}

DEFAULT_TIER_ORDER = ["S1", "S4", "M2", "M3"]

# (scene_id, scene_dir, scene_prefix). scene_id is the short key users pass
# via --scenes; scene_prefix is the YAML filename stem before the tier suffix.
SCENES = [
    ("cube_stack",       "env_configs/cube_stack",       "franka_robosuite_cube_stack"),
    ("two_arm_lift",     "env_configs/two_arm_lift",     "franka_robosuite_two_arm_lift"),
    ("two_arm_handover", "env_configs/two_arm_handover", "two_arm_handover"),
    ("spill_wipe",       "env_configs/spill_wipe",       "franka_robosuite_spill_wipe"),
]

# Per-(scene_prefix, tier) suffix overrides where the on-disk filename does
# not follow the default ``{prefix}_{tier_suffix}.yaml`` pattern.
SUFFIX_OVERRIDE: dict[tuple[str, str], str] = {
    ("franka_robosuite_two_arm_lift", "S4"): "reduced_exampleless",
    ("two_arm_handover",              "S4"): "reduced_exampleless",
}


def _build_yaml_path(scene_dir: str, scene_prefix: str, tier: str) -> str:
    suffix = SUFFIX_OVERRIDE.get((scene_prefix, tier), TIER_SUFFIX[tier])
    return os.path.join(scene_dir, f"{scene_prefix}_{suffix}.yaml")


# ---------------------------------------------------------------------------
# Job assembly
# ---------------------------------------------------------------------------

def _build_jobs(args: argparse.Namespace) -> list[dict]:
    """Produce the full ordered (model, yaml) job list.

    Order: outer loop over models, then scenes, then tiers (S1 -> S4 -> M2 -> M3),
    so all 16 yamls of one model finish before the next model starts.
    """
    scene_filter = set(args.scenes) if args.scenes else None
    tier_filter = list(args.tiers) if args.tiers else DEFAULT_TIER_ORDER

    jobs: list[dict] = []
    for model in args.models:
        for scene_id, scene_dir, scene_prefix in SCENES:
            if scene_filter is not None and scene_id not in scene_filter:
                continue
            for tier in tier_filter:
                yaml_path = _build_yaml_path(scene_dir, scene_prefix, tier)
                jobs.append({
                    "model": model,
                    "scene_id": scene_id,
                    "tier": tier,
                    "yaml_path": yaml_path,
                    "yaml_stem": Path(yaml_path).stem,
                })
    return jobs


def _validate_yamls(jobs: list[dict]) -> None:
    """Fail fast if any expected YAML is missing on disk."""
    missing = [j["yaml_path"] for j in jobs if not os.path.isfile(j["yaml_path"])]
    if missing:
        raise FileNotFoundError(
            "The following sweep YAMLs do not exist (check SCENES/TIER_SUFFIX/"
            "SUFFIX_OVERRIDE in batch_run.py):\n  - " + "\n  - ".join(missing)
        )


def _build_cmd(job: dict, args: argparse.Namespace, per_launch_workers: int) -> list[str]:
    cmd = [
        "uv", "run", "--no-sync", "--active",
        "capx/envs/launch.py",
        "--config-path", job["yaml_path"],
        "--model", job["model"],
        "--total-trials", str(args.total_trials),
        "--num-workers", str(per_launch_workers),
        "--reasoning-effort", args.reasoning_effort,
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--server-url", args.server_url,
        "--visual-differencing-model", args.visual_differencing_model,
    ]
    if args.exp_name:
        cmd += ["--exp-name", args.exp_name]
    if args.overwrite:
        cmd += ["--overwrite"]
    return cmd


# ---------------------------------------------------------------------------
# Process pool
# ---------------------------------------------------------------------------

def _safe_model_dir(model: str) -> str:
    return model.replace("/", "_")


def _resolve_log_path(log_root: Path, exp_name: str | None, job: dict) -> Path:
    exp_dir = log_root / (exp_name if exp_name else "noexp")
    return exp_dir / _safe_model_dir(job["model"]) / f"{job['yaml_stem']}.log"


def _run_one(
    job: dict,
    cmd: list[str],
    log_path: Path,
    summary_path: Path,
    summary_lock: threading.Lock,
    sem: threading.Semaphore,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with sem:
        started_at = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        with open(log_path, "w") as f:
            f.write(f"+ {shlex.join(cmd)}\n")
            f.flush()
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            f.write(f"\n[exit_code={rc} elapsed_s={elapsed:.1f}]\n")
        ended_at = datetime.now().isoformat(timespec="seconds")
        with summary_lock:
            new_file = not summary_path.exists()
            with open(summary_path, "a", newline="") as sf:
                w = csv.writer(sf)
                if new_file:
                    w.writerow(["yaml", "model", "scene", "tier", "exit_code",
                                "elapsed_s", "started_at", "ended_at", "log"])
                w.writerow([
                    job["yaml_path"], job["model"], job["scene_id"], job["tier"],
                    rc, f"{elapsed:.1f}", started_at, ended_at, str(log_path),
                ])
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{status}] {job['model']} :: {job['yaml_stem']}  ({elapsed:.0f}s)  -> {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch driver for CaP-Bench scene x tier sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model identifiers passed straight to launch.py --model.")
    p.add_argument("--exp-name", default=None,
                   help="Forwarded as --exp-name to launch.py; also segments the log dir.")
    p.add_argument("--total-trials", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=12,
                   help="GLOBAL ceiling on concurrent trial workers across all "
                        "active launches. Each launch gets floor(num_workers / "
                        "max_parallel_launches), min 1.")
    p.add_argument("--max-parallel-launches", type=int, default=2,
                   help="K: at most this many launch.py processes run in parallel.")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["off", "minimal", "low", "medium", "high"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=20480)
    p.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    p.add_argument("--visual-differencing-model", default="qwen3-vl-235b-a22b-instruct")
    p.add_argument("--overwrite", action="store_true",
                   help="Pass --overwrite through to launch.py (force rerun even "
                        "if outputs already contain enough trials).")
    p.add_argument("--tiers", nargs="+", default=None, choices=list(TIER_SUFFIX.keys()),
                   help=f"Subset of tiers to run. Default: {DEFAULT_TIER_ORDER}.")
    p.add_argument("--scenes", nargs="+", default=None,
                   choices=[s[0] for s in SCENES],
                   help="Subset of scene ids. Default: all four.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned command list and exit without launching.")
    p.add_argument("--log-dir", default="logs/batch_run",
                   help="Root directory for per-job stdout logs and summary.csv.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_parallel_launches < 1:
        raise ValueError("--max-parallel-launches must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    per_launch_workers = max(1, args.num_workers // args.max_parallel_launches)
    leftover = args.num_workers - per_launch_workers * args.max_parallel_launches

    jobs = _build_jobs(args)
    if not jobs:
        print("No jobs to run after applying --tiers/--scenes filters.")
        return

    _validate_yamls(jobs)

    print("=" * 78)
    print(f"batch_run: {len(jobs)} job(s)")
    print(f"  models               : {args.models}")
    print(f"  scenes               : {args.scenes or [s[0] for s in SCENES]}")
    print(f"  tiers                : {args.tiers or DEFAULT_TIER_ORDER}")
    print(f"  total_trials/job     : {args.total_trials}")
    print(f"  num_workers (global) : {args.num_workers}")
    print(f"  max_parallel_launches: {args.max_parallel_launches}")
    print(f"  per_launch_workers   : {per_launch_workers}"
          + (f"  (leftover {leftover} worker slots unused)" if leftover else ""))
    print(f"  overwrite            : {args.overwrite}")
    print(f"  log_dir              : {args.log_dir}")
    print("=" * 78)

    log_root = Path(args.log_dir)
    summary_path = log_root / (args.exp_name if args.exp_name else "noexp") / "summary.csv"

    plan_lines = []
    for i, job in enumerate(jobs, 1):
        cmd = _build_cmd(job, args, per_launch_workers)
        log_path = _resolve_log_path(log_root, args.exp_name, job)
        plan_lines.append((i, job, cmd, log_path))
        print(f"[{i:02d}/{len(jobs)}] {job['model']} :: {job['scene_id']}/{job['tier']}  "
              f"-> {job['yaml_path']}")
        print(f"        log: {log_path}")
        print(f"        cmd: {shlex.join(cmd)}")

    if args.dry_run:
        print("\n--dry-run set, exiting without launching.")
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sem = threading.Semaphore(args.max_parallel_launches)
    summary_lock = threading.Lock()
    threads: list[threading.Thread] = []

    print(f"\nLaunching {len(plan_lines)} jobs (up to {args.max_parallel_launches} in parallel)...\n")
    overall_t0 = time.time()
    try:
        for _i, job, cmd, log_path in plan_lines:
            t = threading.Thread(
                target=_run_one,
                args=(job, cmd, log_path, summary_path, summary_lock, sem),
                daemon=False,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received; waiting for in-flight launches to exit...")
        for t in threads:
            t.join()
        sys.exit(1)

    elapsed = time.time() - overall_t0
    print(f"\nbatch_run done in {elapsed/60:.1f} min. Summary: {summary_path}")


if __name__ == "__main__":
    main()

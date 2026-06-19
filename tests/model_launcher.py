"""Standalone SAM3 GPU-memory stress launcher.

Spawns N independent SAM3 servers (each loading its own model into VRAM) on
different ports and keeps them alive until Ctrl-C. Intended purely for
GPU-memory / concurrency stress testing -- it deliberately does NOT depend on
``capx.serving.launch_servers`` or any profile/YAML logic.

One-shot run:

    uv run --no-sync --active python tests/sam3_stress_launcher.py \
        --num-instances 4 --gpus 0 --base-port 18114 \
        --log-dir logs/sam3_stress

Multi-GPU round-robin:

    uv run --no-sync --active python tests/sam3_stress_launcher.py \
        --num-instances 12 --gpus 0,1,2,3

Inspect the plan without spawning anything:

    uv run --no-sync --active python tests/sam3_stress_launcher.py \
        --num-instances 8 --gpus 0 --dry-run

Safety features:
  * Hard upper bound on instance count (``--max-instances``, default 32).
  * Pre-flight free-port check on every target port; aborts before spawning
    anything if a port is busy.
  * Pre-flight GPU id sanity (rejects negative ids; warns when a requested
    id is not visible to ``nvidia-smi``).
  * Fail-fast on subprocess startup death or unexpected later exit.
  * Graceful shutdown on SIGINT / SIGTERM (SIGTERM -> wait grace -> SIGKILL).
  * Per-instance log file (``sam3_p{port}_gpu{gpu}.log``) so logs from many
    servers don't interleave.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sam3_stress: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sam3_stress")


@dataclass
class Args:
    """Standalone SAM3 stress launcher CLI args."""

    num_instances: int = 2
    """Number of SAM3 server processes to spawn."""

    base_port: int = 18114
    """First port; instance i uses ``base_port + i``."""

    host: str = "127.0.0.1"
    """Bind host. Keep loopback unless you know what you're doing."""

    gpus: str = "0"
    """Comma-separated CUDA device ids. Round-robin assigned. e.g. '0' or '0,1,2'."""

    checkpoint_path: str | None = None
    """Optional explicit SAM3 checkpoint. If None, falls through to env var
    ``SAM3_CHECKPOINT_PATH`` and finally to HF download (handled inside
    ``capx.serving.launch_sam3_server``)."""

    log_dir: str = "logs/sam3_stress"
    """Per-instance log directory (created if missing)."""

    dry_run: bool = False
    """Print the launch plan and exit without spawning anything."""

    readiness_timeout: float = 600.0
    """Seconds to wait for every instance's TCP port to become connectable.
    SAM3 cold start can be slow when downloading from HF; bump if needed."""

    max_instances: int = 32
    """Hard safety cap. Raise explicitly if you really intend more."""

    shutdown_grace_sec: float = 15.0
    """Seconds to wait for graceful SIGTERM before escalating to SIGKILL."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_gpu_list(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("--gpus is empty")
    ids: list[int] = []
    for p in parts:
        v = int(p)
        if v < 0:
            raise ValueError(f"GPU id must be >= 0, got {v}")
        ids.append(v)
    return ids


def _detect_visible_gpu_count() -> int | None:
    """Best-effort: returns visible GPU count or None if nvidia-smi missing.

    Used only for a *warning* when a requested id is out of range. We don't
    abort on this -- e.g. when ``CUDA_VISIBLE_DEVICES`` is preset, indices
    inside the child are remapped anyway.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return len([ln for ln in result.stdout.strip().splitlines() if ln.strip()])


def _port_free(host: str, port: int) -> bool:
    """True iff we can bind to ``host:port`` right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(0.5)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _tcp_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    """True iff a TCP connection to ``host:port`` succeeds (server is up)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _build_cmd(
    gpu_id: int,
    port: int,
    host: str,
    checkpoint_path: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "capx.serving.launch_sam3_server",
        "--device",
        f"cuda:{gpu_id}",
        "--port",
        str(port),
        "--host",
        host,
    ]
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint-path", checkpoint_path])
    return cmd


# ---------------------------------------------------------------------------
# Instance bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    idx: int
    gpu_id: int
    port: int
    proc: subprocess.Popen[str]
    log_handle: IO[str]
    log_path: Path
    ready: bool = False


def _print_plan(args: Args, plan: list[tuple[int, int, int]]) -> None:
    logger.info("Launch plan (%d instance(s)):", len(plan))
    logger.info("  %-4s %-5s %-5s %s", "idx", "gpu", "port", "url")
    logger.info("  %s", "-" * 36)
    for idx, port, gpu_id in plan:
        logger.info(
            "  %-4d %-5d %-5d http://%s:%d",
            idx,
            gpu_id,
            port,
            args.host,
            port,
        )


def _preflight(args: Args, gpu_ids: list[int]) -> list[tuple[int, int, int]]:
    """Validate args and reserve ports. Returns ``[(idx, port, gpu_id), ...]``."""
    if args.num_instances <= 0:
        raise ValueError(f"--num-instances must be > 0, got {args.num_instances}")
    if args.num_instances > args.max_instances:
        raise ValueError(
            f"--num-instances={args.num_instances} exceeds safety cap "
            f"--max-instances={args.max_instances}; raise --max-instances "
            f"explicitly if intended."
        )
    if args.base_port < 1024 or args.base_port + args.num_instances > 65535:
        raise ValueError(
            f"base_port={args.base_port} with num_instances={args.num_instances} "
            f"is out of the safe TCP port range [1024, 65535]."
        )

    visible = _detect_visible_gpu_count()
    if visible is not None:
        for g in gpu_ids:
            if g >= visible:
                logger.warning(
                    "Requested GPU id %d but nvidia-smi only reports %d device(s); "
                    "assuming CUDA_VISIBLE_DEVICES is set externally.",
                    g,
                    visible,
                )

    plan: list[tuple[int, int, int]] = []
    for i in range(args.num_instances):
        port = args.base_port + i
        gpu_id = gpu_ids[i % len(gpu_ids)]
        if not _port_free(args.host, port):
            raise RuntimeError(
                f"Port {port} on {args.host} is already in use; "
                f"choose another --base-port or stop the conflicting service."
            )
        plan.append((i, port, gpu_id))
    return plan


def _spawn_all(args: Args, plan: list[tuple[int, int, int]]) -> list[Instance]:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[Instance] = []
    try:
        for idx, port, gpu_id in plan:
            cmd = _build_cmd(
                gpu_id=gpu_id,
                port=port,
                host=args.host,
                checkpoint_path=args.checkpoint_path,
            )
            log_path = log_dir / f"sam3_p{port}_gpu{gpu_id}.log"
            fh = open(log_path, "w", buffering=1)  # noqa: SIM115
            logger.info(
                "[SPAWN] idx=%d gpu=%d port=%d log=%s",
                idx,
                gpu_id,
                port,
                log_path,
            )
            logger.info("        cmd: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            started.append(
                Instance(
                    idx=idx,
                    gpu_id=gpu_id,
                    port=port,
                    proc=proc,
                    log_handle=fh,
                    log_path=log_path,
                )
            )
    except BaseException:
        _terminate_all(started, grace_sec=5.0)
        raise
    return started


def _wait_for_ready(args: Args, instances: list[Instance]) -> None:
    deadline = time.monotonic() + args.readiness_timeout
    pending = list(instances)
    interval = 1.0
    max_interval = 5.0

    while pending and time.monotonic() < deadline:
        still: list[Instance] = []
        for inst in pending:
            rc = inst.proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"SAM3 idx={inst.idx} (port={inst.port}, gpu={inst.gpu_id}) "
                    f"exited during startup with code={rc}; see {inst.log_path}"
                )
            if _tcp_ready(args.host, inst.port):
                inst.ready = True
                logger.info(
                    "[READY] idx=%d gpu=%d port=%d",
                    inst.idx,
                    inst.gpu_id,
                    inst.port,
                )
            else:
                still.append(inst)
        pending = still
        if pending:
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            interval = min(interval * 1.5, max_interval)

    if pending:
        names = ", ".join(
            f"idx={i.idx}(port={i.port},gpu={i.gpu_id})" for i in pending
        )
        raise RuntimeError(
            f"Timeout after {args.readiness_timeout:.0f}s waiting for: {names}. "
            f"Inspect logs under {args.log_dir} for the cause."
        )


def _terminate_all(instances: list[Instance], grace_sec: float) -> None:
    if not instances:
        return
    logger.info("Shutting down %d SAM3 instance(s)...", len(instances))

    for inst in instances:
        if inst.proc.poll() is None:
            logger.info(
                "  Terminating idx=%d (pid=%d, port=%d)",
                inst.idx,
                inst.proc.pid,
                inst.port,
            )
            inst.proc.terminate()

    deadline = time.monotonic() + grace_sec
    for inst in instances:
        remain = max(0.1, deadline - time.monotonic())
        try:
            inst.proc.wait(timeout=remain)
        except subprocess.TimeoutExpired:
            logger.warning(
                "  idx=%d (pid=%d) did not exit in %.1fs; sending SIGKILL.",
                inst.idx,
                inst.proc.pid,
                grace_sec,
            )
            inst.proc.kill()
            inst.proc.wait(timeout=5)

    for inst in instances:
        try:
            inst.log_handle.close()
        except OSError as exc:
            logger.warning("Failed to close log %s: %s", inst.log_path, exc)


def _print_status(instances: list[Instance]) -> None:
    logger.info("Final status:")
    logger.info("  %-4s %-5s %-5s %-7s %s", "idx", "gpu", "port", "pid", "status")
    logger.info("  %s", "-" * 40)
    for inst in instances:
        rc = inst.proc.poll()
        if rc is None:
            status = "running"
        elif rc == 0:
            status = "stopped"
        else:
            status = f"exit({rc})"
        logger.info(
            "  %-4d %-5d %-5d %-7d %s",
            inst.idx,
            inst.gpu_id,
            inst.port,
            inst.proc.pid,
            status,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    gpu_ids = _parse_gpu_list(args.gpus)
    plan = _preflight(args, gpu_ids)
    _print_plan(args, plan)

    if args.dry_run:
        logger.info("Dry run; not spawning.")
        return

    instances = _spawn_all(args, plan)

    stop_requested = {"flag": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        if stop_requested["flag"]:
            logger.warning("Second signal received; killing children immediately.")
            for inst in instances:
                if inst.proc.poll() is None:
                    inst.proc.kill()
            return
        stop_requested["flag"] = True
        logger.info(
            "Received %s; will shut down on next tick.",
            signal.Signals(signum).name,
        )

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        _wait_for_ready(args, instances)
        logger.info(
            "All %d instance(s) ready. Holding GPU memory. Press Ctrl-C to stop.",
            len(instances),
        )

        while not stop_requested["flag"]:
            for inst in instances:
                rc = inst.proc.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"SAM3 idx={inst.idx} (port={inst.port}, gpu={inst.gpu_id}) "
                        f"exited unexpectedly with code={rc}; see {inst.log_path}"
                    )
            time.sleep(1.0)
    finally:
        _terminate_all(instances, grace_sec=args.shutdown_grace_sec)
        _print_status(instances)
        logger.info("All SAM3 instances stopped.")


if __name__ == "__main__":
    main(tyro.cli(Args))

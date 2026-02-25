"""Orchestrator for all 24 training runs in the 25_02 sweep.

Manages launching, parallelism, and progress tracking.

Usage:
  python run_sweep.py                          # default settings
  python run_sweep.py --max-length 4096        # explicit
  python run_sweep.py --max-parallel 4         # limit parallelism
  python run_sweep.py --dry-run                # just print what would run
  python run_sweep.py --filter-only            # only run data filtering
  python run_sweep.py --base-model llama8b     # subset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep_25_02.config import (
    BASE_MODELS, TRACE_SOURCES, TASKS,
    DEFAULT_MAX_LENGTH, DEFAULT_LORA_RANK, DEFAULT_LR,
    DEFAULT_BATCH_SIZE, DEFAULT_SCHEDULE,
    TRAINING_BASE, SWEEP_DIR,
    all_run_combos, build_run_name,
)


def run_filter(max_length: int, base_model: str | None = None,
               task: str | None = None, traces: str | None = None) -> bool:
    """Run filter_data.py and return success."""
    cmd = [
        sys.executable, str(SWEEP_DIR / "filter_data.py"),
        "--max-length", str(max_length),
    ]
    if base_model:
        cmd += ["--base-model", base_model]
    if task:
        cmd += ["--task", task]
    if traces:
        cmd += ["--traces", traces]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return result.returncode == 0


def probe_parallelism(max_length: int) -> int:
    """Probe how many concurrent training jobs the cluster can handle.

    Launches 2 tiny test jobs and checks if both get scheduled.
    Returns detected parallelism (default: 2 if probing fails).
    """
    print("\nProbing parallelism...")
    # For now, return a conservative default.
    # Real probing would launch 2 tiny tinker jobs and check scheduling.
    # The user can override with --max-parallel.
    print("  Using default parallelism: 2 (override with --max-parallel)")
    return 2


def launch_training(combo: dict, max_length: int, rank: int, lr: float,
                    batch_size: int, schedule: str) -> subprocess.Popen:
    """Launch a single training run as a subprocess."""
    cmd = [
        sys.executable, str(SWEEP_DIR / "train.py"),
        "--base-model", combo["base_short"],
        "--task", combo["task"],
        "--traces", combo["traces"],
        "--max-length", str(max_length),
        "--lora-rank", str(rank),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--schedule", schedule,
    ]

    log_dir = TRAINING_BASE / combo["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    print(f"  Launching: {combo['name']}")
    print(f"    Log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc


def check_progress(combo: dict) -> dict:
    """Check training progress by reading metrics.jsonl."""
    metrics_path = TRAINING_BASE / combo["name"] / "metrics.jsonl"
    if not metrics_path.exists():
        return {"status": "no_metrics", "step": 0}

    last_step = 0
    last_nll = None
    with open(metrics_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                step = d.get("step", d.get("progress/batch", 0))
                if step > last_step:
                    last_step = step
                if "test/nll" in d:
                    last_nll = d["test/nll"]
            except json.JSONDecodeError:
                continue

    return {"status": "running", "step": last_step, "test_nll": last_nll}


def print_status_table(combos: list[dict], statuses: dict):
    """Print a status table for all runs."""
    print(f"\n{'='*80}")
    print(f"{'Run Name':<45} {'Status':<12} {'Step':>6} {'Test NLL':>10}")
    print(f"{'-'*80}")
    for combo in combos:
        name = combo["name"]
        s = statuses.get(name, {"status": "pending"})
        step = s.get("step", "—")
        nll = s.get("test_nll")
        nll_str = f"{nll:.4f}" if nll is not None else "—"
        print(f"{name:<45} {s['status']:<12} {str(step):>6} {nll_str:>10}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the full 25_02 sweep")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent training jobs (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without launching anything")
    parser.add_argument("--filter-only", action="store_true",
                        help="Only run data filtering, then exit")
    parser.add_argument("--base-model", type=str, default=None,
                        choices=list(BASE_MODELS.keys()),
                        help="Run only for this base model")
    parser.add_argument("--task", type=str, default=None, choices=["math", "code"])
    parser.add_argument("--traces", type=str, default=None,
                        choices=["sonnet", "qwen", "kimi"])
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks (default: 60)")
    args = parser.parse_args()

    combos = all_run_combos(args.max_length, args.lora_rank, args.lr)

    # Apply filters
    if args.base_model:
        combos = [c for c in combos if c["base_short"] == args.base_model]
    if args.task:
        combos = [c for c in combos if c["task"] == args.task]
    if args.traces:
        combos = [c for c in combos if c["traces"] == args.traces]

    print(f"Sweep 25_02: {len(combos)} training runs")
    print(f"  max_length={args.max_length}, rank={args.lora_rank}, lr={args.lr}")

    if args.dry_run:
        print("\nDRY RUN — would launch:")
        for c in combos:
            print(f"  {c['name']}")
        return

    # Phase 1: Filter data
    print("\n" + "="*60)
    print("PHASE 1: Data filtering")
    print("="*60)
    ok = run_filter(args.max_length, args.base_model, args.task, args.traces)
    if not ok:
        print("ERROR: Filtering failed. Aborting.")
        sys.exit(1)

    if args.filter_only:
        print("\n--filter-only: stopping after filtering.")
        return

    # Phase 2: Determine parallelism
    max_parallel = args.max_parallel or probe_parallelism(args.max_length)
    print(f"\nMax parallel jobs: {max_parallel}")

    # Phase 3: Launch training runs
    print("\n" + "="*60)
    print("PHASE 2: Training")
    print("="*60)

    # Track status
    status_path = SWEEP_DIR / "sweep_status.json"
    statuses = {}
    active_procs: dict[str, subprocess.Popen] = {}
    pending = list(combos)
    completed = []

    t0 = time.time()

    while pending or active_procs:
        # Launch new jobs if we have capacity
        while pending and len(active_procs) < max_parallel:
            combo = pending.pop(0)
            name = combo["name"]

            # Skip if already completed (e.g. from a previous run)
            ckpt_file = TRAINING_BASE / name / "checkpoints.jsonl"
            if ckpt_file.exists():
                print(f"  Skipping {name} (already has checkpoints)")
                statuses[name] = {"status": "completed_prior"}
                completed.append(combo)
                continue

            proc = launch_training(
                combo, args.max_length, args.lora_rank, args.lr,
                args.batch_size, args.schedule,
            )
            active_procs[name] = proc
            statuses[name] = {"status": "running", "step": 0, "pid": proc.pid}

        # Check active processes
        finished = []
        for name, proc in active_procs.items():
            ret = proc.poll()
            if ret is not None:
                finished.append(name)
                if ret == 0:
                    statuses[name] = {"status": "completed", "return_code": ret}
                    print(f"  COMPLETED: {name}")
                else:
                    statuses[name] = {"status": "failed", "return_code": ret}
                    print(f"  FAILED: {name} (exit code {ret})")

        for name in finished:
            del active_procs[name]
            combo = next(c for c in combos if c["name"] == name)
            completed.append(combo)

        # Update progress for active jobs
        for name in active_procs:
            combo = next(c for c in combos if c["name"] == name)
            progress = check_progress(combo)
            statuses[name].update(progress)

        # Print status
        if active_procs or finished:
            elapsed = time.time() - t0
            n_done = len(completed)
            n_active = len(active_procs)
            n_pending = len(pending)
            print(f"  [{elapsed:.0f}s] Done: {n_done}, Active: {n_active}, Pending: {n_pending}")

        # Save status
        with open(status_path, "w") as f:
            json.dump(statuses, f, indent=2)

        if active_procs:
            time.sleep(args.poll_interval)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    n_completed = sum(1 for s in statuses.values() if s.get("status") in ("completed", "completed_prior"))
    n_failed = sum(1 for s in statuses.values() if s.get("status") == "failed")
    print(f"  Completed: {n_completed}")
    print(f"  Failed: {n_failed}")
    print(f"  Status saved to: {status_path}")

    print_status_table(combos, statuses)


if __name__ == "__main__":
    main()

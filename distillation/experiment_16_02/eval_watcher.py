"""Experiment 16/02 — Checkpoint eval watcher.

Polls training_runs/*/checkpoints.jsonl for new checkpoints and launches
eval sweeps for each. Results saved to results/{run_name}_step{NNN}/.

Usage:
  PYTHONUNBUFFERED=1 python eval_watcher.py

  # Watch specific training runs only
  PYTHONUNBUFFERED=1 python eval_watcher.py --runs math_default,code_default

  # Override eval settings
  PYTHONUNBUFFERED=1 python eval_watcher.py --max-problems 50 --max-tokens 8192
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent))

from experiment_16_02.config import (
    MODEL_NAME, INFERENCE_MAX_TOKENS, EVAL_DATASETS,
    RESULTS_DIR, TRAINING_DIR,
)

MAX_CONCURRENT = 2

DATASET_PARQUETS = {
    "math_500": "results_math_500.parquet",
    "aime": "results_aime.parquet",
    "humanevalplus": "results_humanevalplus.parquet",
    "mbppplus": "results_mbppplus.parquet",
    "kodcode_500": "results_kodcode_500.parquet",
    "codeforces_500": "results_codeforces_500.parquet",
    "livecodebench_v5": "results_livecodebench_v5.parquet",
    "omni_math": "results_omni_math.parquet",
}


def read_checkpoints(ckpt_file: Path) -> list[dict]:
    """Read checkpoints from a JSONL file."""
    if not ckpt_file.exists():
        return []
    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return checkpoints


def get_missing_datasets(results_name: str, datasets: list[str]) -> list[str]:
    """Check which dataset results are missing for a given results_name."""
    results_dir = RESULTS_DIR / results_name
    missing = []
    for ds in datasets:
        parquet = DATASET_PARQUETS.get(ds)
        if parquet and not (results_dir / parquet).exists():
            missing.append(ds)
    return missing


def find_training_runs(run_names: list[str] | None = None) -> list[str]:
    """Find all training run directories with checkpoints.jsonl."""
    runs = []
    if not TRAINING_DIR.exists():
        return runs
    for d in sorted(TRAINING_DIR.iterdir()):
        if not d.is_dir():
            continue
        if run_names and d.name not in run_names:
            continue
        if (d / "checkpoints.jsonl").exists():
            runs.append(d.name)
    return runs


def collect_pending_evals(run_names: list[str] | None, datasets: list[str]) -> list[dict]:
    """Find all checkpoints that need evaluation."""
    pending = []
    runs = find_training_runs(run_names)

    for run_name in runs:
        ckpt_file = TRAINING_DIR / run_name / "checkpoints.jsonl"
        checkpoints = read_checkpoints(ckpt_file)

        for ckpt in checkpoints:
            step = ckpt.get("batch", -1)
            sampler_path = ckpt.get("sampler_path")
            if step < 0 or not sampler_path:
                continue

            results_name = f"{run_name}_step{step:03d}"
            missing = get_missing_datasets(results_name, datasets)
            if missing:
                pending.append({
                    "results_name": results_name,
                    "sampler_path": sampler_path,
                    "run_name": run_name,
                    "step": step,
                    "datasets": missing,
                })

    # Sort by step ascending, then run name
    pending.sort(key=lambda j: (j["step"], j["run_name"]))
    return pending


def launch_eval(job: dict, max_tokens: int, max_problems: int | None,
                fast: bool = True) -> subprocess.Popen:
    """Launch an eval subprocess for a checkpoint."""
    log_dir = TRAINING_DIR / job["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_step{job['step']:03d}.log"

    cmd = [
        sys.executable, str(EXPERIMENT_DIR / "evaluate.py"),
        "checkpoint",
        "--sampler-path", job["sampler_path"],
        "--model-name", MODEL_NAME,
        "--results-name", job["results_name"],
        "--max-tokens", str(max_tokens),
    ]
    if fast:
        cmd.append("--fast")
    else:
        cmd.extend(["--datasets", ",".join(job["datasets"])])
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])

    print(f"  LAUNCH: {job['results_name']} ({', '.join(job['datasets'])})")
    print(f"    log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(EXPERIMENT_DIR.parent),
        )
    return proc


def main():
    parser = argparse.ArgumentParser(description="Experiment 16/02: Checkpoint eval watcher")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated training run names to watch (default: all)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent eval processes (default: {MAX_CONCURRENT})")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Fast mode: math_500 (100) + mbppplus (100) (default: True)")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: all 4 datasets, full size")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit problems per dataset (default: 100 in fast mode)")
    parser.add_argument("--max-tokens", type=int, default=INFERENCE_MAX_TOKENS,
                        help=f"Max tokens for generation (default: {INFERENCE_MAX_TOKENS})")
    parser.add_argument("--datasets", type=str, default=None,
                        help=f"Comma-separated datasets (default: {','.join(EVAL_DATASETS)})")
    args = parser.parse_args()

    run_names = args.runs.split(",") if args.runs else None
    fast = args.fast and not args.full
    if args.datasets:
        datasets = args.datasets.split(",")
    elif fast:
        datasets = ["math_500", "mbppplus"]
    else:
        datasets = EVAL_DATASETS

    print(f"{'='*60}")
    print(f"Experiment 16/02 — Checkpoint Eval Watcher")
    print(f"  Watching: {run_names or 'all runs'}")
    print(f"  Mode: {'fast (math_500:100 + mbppplus:100)' if fast else 'full'}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"{'='*60}\n")

    active: list[tuple[str, subprocess.Popen]] = []
    seen: set[str] = set()
    completed = 0
    failed = 0

    while True:
        # Reap finished processes
        still_running = []
        for name, proc in active:
            rc = proc.poll()
            if rc is None:
                still_running.append((name, proc))
            elif rc == 0:
                completed += 1
                print(f"  DONE: {name}")
            else:
                failed += 1
                print(f"  FAILED: {name} (rc={rc})")
        active = still_running

        # Find new pending evals
        pending = collect_pending_evals(run_names, datasets)
        new_jobs = [j for j in pending
                    if j["results_name"] not in seen
                    and j["results_name"] not in {n for n, _ in active}]

        # Launch new evals up to concurrency limit
        for job in new_jobs:
            if len(active) >= args.max_concurrent:
                break
            proc = launch_eval(job, args.max_tokens, args.max_problems, fast=fast)
            active.append((job["results_name"], proc))
            seen.add(job["results_name"])
            time.sleep(1)

        # Check if any training is still running (no "final" checkpoint yet)
        training_alive = False
        for run_name in find_training_runs(run_names):
            ckpt_file = TRAINING_DIR / run_name / "checkpoints.jsonl"
            checkpoints = read_checkpoints(ckpt_file)
            has_final = any(c.get("name") == "final" for c in checkpoints)
            if not has_final:
                training_alive = True

        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"active: {len(active)}, done: {completed}, failed: {failed}, "
            f"training: {'running' if training_alive else 'complete'}"
        )

        # Exit when training done, no active evals, and nothing pending
        if not training_alive and not active:
            remaining = collect_pending_evals(run_names, datasets)
            if not remaining:
                break
            for job in remaining:
                if len(active) >= args.max_concurrent:
                    break
                if job["results_name"] not in seen:
                    proc = launch_eval(job, args.max_tokens, args.max_problems, fast=fast)
                    active.append((job["results_name"], proc))
                    seen.add(job["results_name"])
                    time.sleep(1)
            if not active:
                break

        time.sleep(30)

    # Final summary
    print(f"\n{'='*60}")
    print(f"EVAL WATCHER COMPLETE")
    print(f"  Completed: {completed}, Failed: {failed}")
    print(f"{'='*60}")

    # Print results table
    print(f"\nResults:")
    import pandas as pd
    for run_name in find_training_runs(run_names):
        ckpt_file = TRAINING_DIR / run_name / "checkpoints.jsonl"
        for ckpt in read_checkpoints(ckpt_file):
            step = ckpt.get("batch", -1)
            if step < 0:
                continue
            results_name = f"{run_name}_step{step:03d}"
            results_dir = RESULTS_DIR / results_name
            parts = []
            for ds in datasets:
                pq = DATASET_PARQUETS.get(ds)
                if not pq:
                    continue
                pq_path = results_dir / pq
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                    col = "passed" if "passed" in df.columns else "correct"
                    if col in df.columns:
                        acc = df[col].mean() * 100
                        parts.append(f"{ds}={acc:.1f}%")
            if parts:
                print(f"  {results_name}: {', '.join(parts)}")
            else:
                print(f"  {results_name}: pending")


if __name__ == "__main__":
    main()

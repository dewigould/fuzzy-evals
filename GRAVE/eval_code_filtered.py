"""Eval watcher for filtered code training runs.

Monitors base_kodcode_filtered and base_kodcode_4k_filtered for new checkpoints,
evaluates each on kodcode_500, codeforces_500, livecodebench_v5 with max_tokens=16384.

Usage:
  PYTHONUNBUFFERED=1 python eval_code_filtered.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from config import TRAINING_BASE, RESULTS_BASE

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"
MAX_CONCURRENT = 4
MAX_TOKENS = 16384
MAX_PROBLEMS = 100

DATASETS = "kodcode_500,codeforces_500,livecodebench_v5"
DATASET_PARQUETS = {
    "kodcode_500": "results_kodcode_500.parquet",
    "codeforces_500": "results_codeforces_500.parquet",
    "livecodebench_v5": "results_livecodebench_v5.parquet",
}

TRAINING_RUNS = [
    {"task": "base_kodcode_filtered", "config": "A_r32_high_lr"},
    {"task": "base_kodcode_4k_filtered", "config": "A_r32_high_lr"},
]


def read_checkpoints(ckpt_file: Path) -> list[dict]:
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


def get_missing_datasets(results_name: str) -> list[str]:
    results_dir = RESULTS_BASE / results_name
    missing = []
    for ds_name, parquet_name in DATASET_PARQUETS.items():
        if not (results_dir / parquet_name).exists():
            missing.append(ds_name)
    return missing


def collect_pending_evals() -> list[dict]:
    pending = []
    for run in TRAINING_RUNS:
        ckpt_file = Path(TRAINING_BASE) / run["task"] / run["config"] / "checkpoints.jsonl"
        checkpoints = read_checkpoints(ckpt_file)
        for ckpt in checkpoints:
            step = ckpt.get("batch", -1)
            sampler_path = ckpt.get("sampler_path")
            if step < 0 or not sampler_path:
                continue
            results_name = f"{run['task']}_{run['config']}_step{step:03d}"
            missing = get_missing_datasets(results_name)
            if missing:
                pending.append({
                    "results_name": results_name,
                    "sampler_path": sampler_path,
                    "task": run["task"],
                    "config": run["config"],
                    "step": step,
                    "datasets": missing,
                })
    # Interleave 4K and 8K by step
    pending.sort(key=lambda j: (j["step"], j["task"]))
    return pending


def launch_eval(job: dict) -> subprocess.Popen:
    log_dir = Path(TRAINING_BASE) / job["task"] / job["config"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_step{job['step']:03d}.log"

    cmd = [
        sys.executable, "evaluate.py",
        "--sampler-path", job["sampler_path"],
        "--base-model-name", BASE_MODEL_NAME,
        "--results-name", job["results_name"],
        "--datasets", ",".join(job["datasets"]),
        "--max-problems", str(MAX_PROBLEMS),
        "--max-tokens", str(MAX_TOKENS),
    ]

    print(f"  LAUNCH: {job['results_name']} ({', '.join(job['datasets'])})")
    print(f"    log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    return proc


def main():
    print(f"{'='*60}")
    print(f"Filtered Code Eval Watcher")
    print(f"Datasets: {DATASETS}")
    print(f"Max tokens: {MAX_TOKENS}, Max problems: {MAX_PROBLEMS}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"{'='*60}\n")

    active: list[tuple[str, subprocess.Popen]] = []
    seen_checkpoints: set[str] = set()
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
        pending = collect_pending_evals()
        new_jobs = [j for j in pending if j["results_name"] not in seen_checkpoints
                    and j["results_name"] not in {n for n, _ in active}]

        # Launch new evals
        for job in new_jobs:
            if len(active) >= MAX_CONCURRENT:
                break
            proc = launch_eval(job)
            active.append((job["results_name"], proc))
            seen_checkpoints.add(job["results_name"])
            time.sleep(1)

        # Check if training is still running
        training_alive = False
        for run in TRAINING_RUNS:
            ckpt_file = Path(TRAINING_BASE) / run["task"] / run["config"] / "checkpoints.jsonl"
            checkpoints = read_checkpoints(ckpt_file)
            has_final = any(c.get("batch", -1) == 500 for c in checkpoints)
            if not has_final:
                training_alive = True

        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"active: {len(active)}, done: {completed}, failed: {failed}, "
            f"training: {'running' if training_alive else 'complete'}"
        )

        # Exit when training is done, no active evals, and no pending
        if not training_alive and not active:
            remaining = collect_pending_evals()
            if not remaining:
                break
            for job in remaining:
                if len(active) >= MAX_CONCURRENT:
                    break
                if job["results_name"] not in seen_checkpoints:
                    proc = launch_eval(job)
                    active.append((job["results_name"], proc))
                    seen_checkpoints.add(job["results_name"])
                    time.sleep(1)
            if not active:
                break

        time.sleep(30)

    print(f"\n{'='*60}")
    print(f"CODE EVAL WATCHER COMPLETE")
    print(f"  Completed: {completed}, Failed: {failed}")
    print(f"{'='*60}")

    # Print results summary
    print(f"\nResults:")
    for run in TRAINING_RUNS:
        ckpt_file = Path(TRAINING_BASE) / run["task"] / run["config"] / "checkpoints.jsonl"
        for ckpt in read_checkpoints(ckpt_file):
            step = ckpt.get("batch", -1)
            if step < 0:
                continue
            results_name = f"{run['task']}_{run['config']}_step{step:03d}"
            results_dir = RESULTS_BASE / results_name
            parts = []
            for ds, pq in DATASET_PARQUETS.items():
                pq_path = results_dir / pq
                if pq_path.exists():
                    import pandas as pd
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

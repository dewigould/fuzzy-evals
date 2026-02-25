"""Overnight eval orchestrator — runs all pending evals with bounded concurrency.

Waits for 4K training to finish, then runs all missing evals (8K + 4K)
with at most MAX_CONCURRENT eval processes at a time.

Usage:
  PYTHONUNBUFFERED=1 python overnight_eval.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from config import TRAINING_BASE, RESULTS_BASE

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"
MAX_CONCURRENT = 3

ALL_DATASETS_LIST = [
    "math_500", "aime", "omni_math",
    "fuzzy_philosophy", "fuzzy_weird_qs", "fuzzy_futuristic_tech",
]
N_DATASETS = len(ALL_DATASETS_LIST)

# Map dataset names to their parquet filenames
DATASET_TO_PARQUET = {
    "math_500": "results_math_500.parquet",
    "aime": "results_aime.parquet",
    "omni_math": "results_omni_math.parquet",
    "fuzzy_philosophy": "results_fuzzy_philosophy.parquet",
    "fuzzy_weird_qs": "results_fuzzy_weird_qs.parquet",
    "fuzzy_futuristic_tech": "results_fuzzy_futuristic_tech.parquet",
}

# All training run dirs to scan for checkpoints
TRAINING_RUNS = [
    {"task": "base_math", "config": "A_r32_high_lr"},
    {"task": "base_kodcode", "config": "A_r32_high_lr"},
    {"task": "base_math_4k", "config": "A_r32_high_lr"},
    {"task": "base_kodcode_4k", "config": "A_r32_high_lr"},
]


def get_memory_usage_gb() -> tuple[float, float]:
    """Return (usage_gb, limit_gb) from cgroup."""
    try:
        limit = int(open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read().strip())
        usage = int(open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read().strip())
        return usage / 1e9, limit / 1e9
    except Exception:
        return 0.0, 64.0


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
    """Return list of dataset names that still need eval for this checkpoint."""
    results_dir = RESULTS_BASE / results_name
    missing = []
    for ds_name, parquet_name in DATASET_TO_PARQUET.items():
        if not (results_dir / parquet_name).exists():
            missing.append(ds_name)
    return missing


def collect_pending_evals() -> list[dict]:
    """Scan all training runs for checkpoints that need eval."""
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
    return pending


def launch_eval(job: dict) -> subprocess.Popen:
    """Launch a single evaluate.py process for missing datasets only."""
    log_dir = Path(TRAINING_BASE) / job["task"] / job["config"]
    log_dir.mkdir(parents=True, exist_ok=True)
    datasets = job["datasets"]
    # Use a suffix for partial re-runs so we don't clobber existing logs
    log_file = log_dir / f"eval_step{job['step']:03d}.log"

    cmd = [
        sys.executable, "evaluate.py",
        "--sampler-path", job["sampler_path"],
        "--base-model-name", BASE_MODEL_NAME,
        "--results-name", job["results_name"],
        "--datasets", ",".join(datasets),
        "--max-problems", "100",
        "--fuzzy-samples", "3",
    ]

    print(f"  LAUNCH: {job['results_name']} ({len(datasets)} datasets: {', '.join(datasets)})")
    print(f"    log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    return proc


def wait_for_training():
    """Wait for 4K training runs to finish (check if checkpoints.jsonl has final entry)."""
    print("Waiting for 4K training to finish...")
    for run in TRAINING_RUNS:
        if "4k" not in run["task"]:
            continue
        ckpt_file = Path(TRAINING_BASE) / run["task"] / run["config"] / "checkpoints.jsonl"
        while True:
            checkpoints = read_checkpoints(ckpt_file)
            # Training saves 10 checkpoints (step 50..500) plus possibly a 'final'
            has_final = any("final" in c.get("sampler_path", "") for c in checkpoints)
            has_500 = any(c.get("batch", -1) == 500 for c in checkpoints)
            if has_final or has_500 or len(checkpoints) >= 10:
                print(f"  {run['task']}/{run['config']}: done ({len(checkpoints)} checkpoints)")
                break
            usage, limit = get_memory_usage_gb()
            print(f"  [{time.strftime('%H:%M:%S')}] {run['task']}: {len(checkpoints)} checkpoints, "
                  f"mem: {usage:.1f}/{limit:.1f} GB")
            time.sleep(30)


def main():
    print(f"{'='*60}")
    print(f"Overnight Eval Orchestrator")
    print(f"Max concurrent evals: {MAX_CONCURRENT}")
    print(f"{'='*60}\n")

    # Phase 1: Skip — all training already complete

    # Phase 2: Collect all pending evals
    pending = collect_pending_evals()
    print(f"\nTotal pending evals: {len(pending)}")
    for job in pending:
        print(f"  {job['results_name']} — {len(job['datasets'])} missing: {', '.join(job['datasets'])}")

    if not pending:
        print("Nothing to do!")
        return

    # Phase 3: Run evals with bounded concurrency
    print(f"\nStarting evals (max {MAX_CONCURRENT} concurrent)...\n")
    active: list[tuple[str, subprocess.Popen]] = []
    queue = list(pending)
    completed = 0
    failed = 0

    while queue or active:
        # Reap finished processes
        still_running = []
        for name, proc in active:
            rc = proc.poll()
            if rc is None:
                still_running.append((name, proc))
            else:
                if rc == 0:
                    completed += 1
                    print(f"  DONE: {name}")
                else:
                    failed += 1
                    print(f"  FAILED: {name} (rc={rc})")
        active = still_running

        # Check memory before launching new ones
        usage, limit = get_memory_usage_gb()
        mem_ok = usage < (limit - 5.0)  # keep 5GB headroom

        # Launch new evals up to MAX_CONCURRENT
        while queue and len(active) < MAX_CONCURRENT and mem_ok:
            job = queue.pop(0)
            proc = launch_eval(job)
            active.append((job["results_name"], proc))
            # Recheck memory
            time.sleep(2)
            usage, limit = get_memory_usage_gb()
            mem_ok = usage < (limit - 5.0)
            if not mem_ok:
                print(f"  Memory high ({usage:.1f}/{limit:.1f} GB), pausing launches")

        # Status
        usage, limit = get_memory_usage_gb()
        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"active: {len(active)}, queued: {len(queue)}, "
            f"done: {completed}, failed: {failed}, "
            f"mem: {usage:.1f}/{limit:.1f} GB"
        )

        if active or queue:
            time.sleep(30)

    print(f"\n{'='*60}")
    print(f"OVERNIGHT EVAL COMPLETE")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")

    # Final summary
    print(f"\nResults summary:")
    for run in TRAINING_RUNS:
        ckpt_file = Path(TRAINING_BASE) / run["task"] / run["config"] / "checkpoints.jsonl"
        checkpoints = read_checkpoints(ckpt_file)
        for ckpt in checkpoints:
            step = ckpt.get("batch", -1)
            if step < 0:
                continue
            results_name = f"{run['task']}_{run['config']}_step{step:03d}"
            missing = get_missing_datasets(results_name)
            summary_file = RESULTS_BASE / results_name / "eval_summary.json"
            if summary_file.exists():
                summary = json.load(open(summary_file))
                parts = []
                for ds, res in summary.items():
                    if "accuracy" in res:
                        parts.append(f"{ds}={res['accuracy']*100:.0f}%")
                    elif "mean_score" in res:
                        parts.append(f"{ds}={res['mean_score']:.1f}")
                print(f"  {results_name}: {', '.join(parts)}")
            else:
                n_parquets = len(list((RESULTS_BASE / results_name).glob("*.parquet"))) if (RESULTS_BASE / results_name).exists() else 0
                status = f"{n_parquets}/{N_DATASETS} datasets" if n_parquets > 0 else "missing"
                print(f"  {results_name}: {status}")


if __name__ == "__main__":
    main()

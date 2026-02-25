"""Polling watcher for 4K-context base model training checkpoints.

Same as eval_watcher.py but watches base_math_4k/ and base_kodcode_4k/ dirs.
Inference still uses max_tokens=28672.

Usage:
  PYTHONUNBUFFERED=1 python eval_watcher_4k.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from config import TRAINING_BASE

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"

ALL_DATASETS = (
    "math_500,aime,omni_math,"
    "codeforces_500,livecodebench_v5,"
    "fuzzy_philosophy,fuzzy_weird_qs,fuzzy_futuristic_tech"
)

WATCH_TARGETS = [
    {
        "task": "base_math_4k",
        "config": "A_r32_high_lr",
        "ckpt_file": Path(TRAINING_BASE) / "base_math_4k" / "A_r32_high_lr" / "checkpoints.jsonl",
    },
    {
        "task": "base_kodcode_4k",
        "config": "A_r32_high_lr",
        "ckpt_file": Path(TRAINING_BASE) / "base_kodcode_4k" / "A_r32_high_lr" / "checkpoints.jsonl",
    },
]

POLL_INTERVAL = 30


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


def launch_eval(task: str, config: str, step: int, sampler_path: str) -> subprocess.Popen:
    results_name = f"{task}_{config}_step{step:03d}"
    log_dir = Path(TRAINING_BASE) / task / config
    log_file = log_dir / f"eval_step{step:03d}.log"

    cmd = [
        sys.executable, "evaluate.py",
        "--sampler-path", sampler_path,
        "--base-model-name", BASE_MODEL_NAME,
        "--results-name", results_name,
        "--datasets", ALL_DATASETS,
        "--max-problems", "100",
        "--fuzzy-samples", "3",
    ]

    print(f"  Launching eval: {results_name}")
    print(f"    sampler_path: {sampler_path}")
    print(f"    log: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    return proc


def main():
    print(f"{'='*60}")
    print(f"Eval Watcher — Base Model 4K Checkpoints")
    print(f"Polling every {POLL_INTERVAL}s")
    print(f"{'='*60}")
    for target in WATCH_TARGETS:
        print(f"  Watching: {target['ckpt_file']}")
    print()

    seen: dict[str, set[int]] = {}
    for target in WATCH_TARGETS:
        key = f"{target['task']}/{target['config']}"
        seen[key] = set()

    active_procs: list[tuple[str, subprocess.Popen]] = []

    try:
        while True:
            for target in WATCH_TARGETS:
                key = f"{target['task']}/{target['config']}"
                checkpoints = read_checkpoints(target["ckpt_file"])

                for ckpt in checkpoints:
                    step = ckpt.get("batch", -1)
                    sampler_path = ckpt.get("sampler_path")
                    if step < 0 or not sampler_path:
                        continue
                    if step in seen[key]:
                        continue

                    seen[key].add(step)
                    proc = launch_eval(target["task"], target["config"], step, sampler_path)
                    active_procs.append((f"{key}/step{step}", proc))

            still_running = []
            for name, proc in active_procs:
                rc = proc.poll()
                if rc is None:
                    still_running.append((name, proc))
                else:
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"  Eval finished: {name} — {status}")
            active_procs = still_running

            total_seen = sum(len(s) for s in seen.values())
            print(
                f"  [{time.strftime('%H:%M:%S')}] "
                f"checkpoints seen: {total_seen}, "
                f"evals running: {len(active_procs)}"
            )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping watcher...")
        for name, proc in active_procs:
            print(f"  Waiting for {name}...")
            proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()

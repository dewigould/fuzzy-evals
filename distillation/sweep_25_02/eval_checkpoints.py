"""Mid-training checkpoint evaluation for the 25_02 sweep.

Evaluates all checkpoints from a training run on MATH-500 + MBPP+ (100 samples each).
Uses evaluate.py CLI in ad-hoc checkpoint mode.

Usage:
  python eval_checkpoints.py --run llama8b_math_sonnet_4096_r32_lr2e-4
  python eval_checkpoints.py --all
  python eval_checkpoints.py --all --max-problems 50  # quick check
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
    BASE_MODELS, TRAINING_BASE, RESULTS_BASE, DEFAULT_MAX_LENGTH,
    all_run_combos,
)


EVAL_SCRIPT = Path(__file__).parent.parent / "evaluate.py"

# Datasets for mid-training eval
MID_TRAIN_DATASETS = "math_500,mbppplus"
DEFAULT_MAX_PROBLEMS = 100


def get_checkpoints(run_name: str) -> list[dict]:
    """Read checkpoints.jsonl for a training run."""
    ckpt_file = TRAINING_BASE / run_name / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []
    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return checkpoints


def get_base_model_for_run(run_name: str) -> str | None:
    """Extract the base model HF name from a run name."""
    for short, full in BASE_MODELS.items():
        if run_name.startswith(short + "_"):
            return full
    return None


def get_no_think_datasets(run_name: str) -> str:
    """Determine which datasets should NOT use think_prefix.

    Math-distilled models: disable think_prefix on math evals (native thinking)
    Code-distilled models: disable think_prefix on code evals (native thinking)
    """
    if "_math_" in run_name:
        return "math_500"
    elif "_code_" in run_name:
        return "mbppplus"
    return ""


def eval_checkpoint(run_name: str, checkpoint: dict,
                    max_problems: int = DEFAULT_MAX_PROBLEMS,
                    max_tokens: int = DEFAULT_MAX_LENGTH) -> bool:
    """Evaluate a single checkpoint. Returns True on success."""
    sampler_path = checkpoint.get("sampler_path")
    step = checkpoint.get("batch", "unknown")
    if not sampler_path:
        print(f"  Skipping checkpoint (no sampler_path): {checkpoint}")
        return False

    results_name = f"{run_name}_step{step}"
    base_model = get_base_model_for_run(run_name)
    no_think = get_no_think_datasets(run_name)

    # Check if already evaluated
    results_dir = RESULTS_BASE / results_name
    math_parquet = results_dir / "results_math_500.parquet"
    mbpp_parquet = results_dir / "results_mbppplus.parquet"
    if math_parquet.exists() and mbpp_parquet.exists():
        print(f"  Step {step}: already evaluated, skipping")
        return True

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--sampler-path", sampler_path,
        "--results-name", results_name,
        "--results-base", str(RESULTS_BASE),
        "--datasets", MID_TRAIN_DATASETS,
        "--max-problems", str(max_problems),
        "--max-tokens", str(max_tokens),
    ]
    if base_model:
        cmd += ["--base-model-name", base_model]
    if no_think:
        cmd += ["--no-think-prefix-datasets", no_think]

    print(f"  Evaluating step {step}: {sampler_path}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=False,
    )
    elapsed = time.time() - t0
    print(f"  Step {step} eval {'OK' if result.returncode == 0 else 'FAILED'} ({elapsed:.0f}s)")
    return result.returncode == 0


def eval_run(run_name: str, max_problems: int = DEFAULT_MAX_PROBLEMS,
             max_tokens: int = DEFAULT_MAX_LENGTH):
    """Evaluate all checkpoints for a training run."""
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoints: {run_name}")
    print(f"{'='*60}")

    checkpoints = get_checkpoints(run_name)
    if not checkpoints:
        print(f"  No checkpoints found for {run_name}")
        return

    print(f"  Found {len(checkpoints)} checkpoints")

    results = []
    for ckpt in checkpoints:
        ok = eval_checkpoint(run_name, ckpt, max_problems, max_tokens)
        results.append({"step": ckpt.get("batch"), "success": ok})

    n_ok = sum(1 for r in results if r["success"])
    print(f"\n  {run_name}: {n_ok}/{len(results)} checkpoints evaluated successfully")


def main():
    parser = argparse.ArgumentParser(description="Evaluate training checkpoints")
    parser.add_argument("--run", type=str, default=None,
                        help="Evaluate a specific run by name")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all completed runs")
    parser.add_argument("--max-problems", type=int, default=DEFAULT_MAX_PROBLEMS,
                        help=f"Problems per dataset (default: {DEFAULT_MAX_PROBLEMS})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max tokens for generation (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Max length used in sweep (for finding runs)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Filter to runs for this base model")
    parser.add_argument("--task", type=str, default=None, choices=["math", "code"])
    args = parser.parse_args()

    if args.run:
        eval_run(args.run, args.max_problems, args.max_tokens)
    elif args.all:
        combos = all_run_combos(args.max_length)
        if args.base_model:
            combos = [c for c in combos if c["base_short"] == args.base_model]
        if args.task:
            combos = [c for c in combos if c["task"] == args.task]

        # Only evaluate runs that have checkpoints
        runs_with_ckpts = []
        for combo in combos:
            ckpts = get_checkpoints(combo["name"])
            if ckpts:
                runs_with_ckpts.append(combo)

        print(f"Found {len(runs_with_ckpts)} runs with checkpoints")

        for combo in runs_with_ckpts:
            eval_run(combo["name"], args.max_problems, args.max_tokens)
    else:
        print("Specify --run NAME or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()

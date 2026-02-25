"""Full evaluation for selected checkpoints on all 6 benchmarks.

Run after checkpoint selection from mid-training eval results.
Evaluates on: MATH-500, OmniMath(500), AIME, MBPP+(500), HumanEval+(500), LiveCodeBench(500)

Think prefix logic for distilled models:
- Math-distilled on math evals: think_prefix=False (learned natively)
- Math-distilled on code evals: think_prefix=True (cross-domain)
- Code-distilled on code evals: think_prefix=False (learned natively)
- Code-distilled on math evals: think_prefix=True (cross-domain)

Usage:
  python eval_full.py --run llama8b_math_sonnet_4096_r32_lr2e-4 --step 300
  python eval_full.py --runs-file selected_checkpoints.json
  python eval_full.py --max-problems 100  # quick check
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
)


EVAL_SCRIPT = Path(__file__).parent.parent / "evaluate.py"

MATH_DATASETS = "math_500,omni_math,aime"
CODE_DATASETS = "mbppplus,humanevalplus,livecodebench_v5"
ALL_DATASETS = f"{MATH_DATASETS},{CODE_DATASETS}"


def get_no_think_datasets(run_name: str) -> str:
    """Determine no_think_prefix_datasets based on training task.

    Math-distilled: no think_prefix on math evals (native), use on code (cross-domain)
    Code-distilled: no think_prefix on code evals (native), use on math (cross-domain)
    """
    if "_math_" in run_name:
        return MATH_DATASETS
    elif "_code_" in run_name:
        return CODE_DATASETS
    return ""


def get_base_model_for_run(run_name: str) -> str | None:
    """Extract the base model HF name from a run name."""
    for short, full in BASE_MODELS.items():
        if run_name.startswith(short + "_"):
            return full
    return None


def find_checkpoint(run_name: str, step: int) -> str | None:
    """Find the sampler_path for a specific step."""
    ckpt_file = TRAINING_BASE / run_name / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None
    with open(ckpt_file) as f:
        for line in f:
            try:
                ckpt = json.loads(line)
                if ckpt.get("batch") == step and "sampler_path" in ckpt:
                    return ckpt["sampler_path"]
            except json.JSONDecodeError:
                continue
    return None


def eval_checkpoint_full(run_name: str, step: int, sampler_path: str,
                         max_tokens: int, max_problems: int | None,
                         datasets: str = ALL_DATASETS) -> bool:
    """Run full evaluation on a checkpoint."""
    results_name = f"{run_name}_step{step}"
    base_model = get_base_model_for_run(run_name)
    no_think = get_no_think_datasets(run_name)

    print(f"\n{'='*60}")
    print(f"Full eval: {run_name} step {step}")
    print(f"  Sampler: {sampler_path}")
    print(f"  Results: {RESULTS_BASE / results_name}")
    print(f"  No think prefix: {no_think}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--sampler-path", sampler_path,
        "--results-name", results_name,
        "--results-base", str(RESULTS_BASE),
        "--datasets", datasets,
        "--max-tokens", str(max_tokens),
    ]
    if base_model:
        cmd += ["--base-model-name", base_model]
    if no_think:
        cmd += ["--no-think-prefix-datasets", no_think]
    if max_problems is not None:
        cmd += ["--max-problems", str(max_problems)]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=False,
    )
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n{run_name} step {step}: {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full evaluation of selected checkpoints")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to evaluate")
    parser.add_argument("--sampler-path", type=str, default=None,
                        help="Direct sampler path (overrides --step lookup)")
    parser.add_argument("--runs-file", type=str, default=None,
                        help="JSON file with list of {run_name, step} to evaluate")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max tokens (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit problems per dataset")
    parser.add_argument("--datasets", type=str, default=ALL_DATASETS,
                        help="Override datasets (comma-separated)")
    args = parser.parse_args()

    runs_to_eval = []

    if args.runs_file:
        with open(args.runs_file) as f:
            runs_to_eval = json.load(f)
    elif args.run and args.step is not None:
        sampler = args.sampler_path or find_checkpoint(args.run, args.step)
        if not sampler:
            print(f"ERROR: No checkpoint found for {args.run} step {args.step}")
            sys.exit(1)
        runs_to_eval = [{"run_name": args.run, "step": args.step, "sampler_path": sampler}]
    elif args.run and args.sampler_path:
        runs_to_eval = [{"run_name": args.run, "step": 0, "sampler_path": args.sampler_path}]
    else:
        print("Specify --run + --step, --run + --sampler-path, or --runs-file")
        sys.exit(1)

    t0 = time.time()
    results = {}
    for entry in runs_to_eval:
        run_name = entry["run_name"]
        step = entry["step"]
        sampler = entry.get("sampler_path") or find_checkpoint(run_name, step)
        if not sampler:
            print(f"  Skipping {run_name} step {step}: no sampler_path found")
            results[f"{run_name}_step{step}"] = False
            continue
        ok = eval_checkpoint_full(
            run_name, step, sampler,
            args.max_tokens, args.max_problems, args.datasets,
        )
        results[f"{run_name}_step{step}"] = ok

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()

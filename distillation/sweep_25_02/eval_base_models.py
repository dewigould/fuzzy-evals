"""Evaluate all 4 base models on the full eval suite (6 datasets).

Datasets: MATH-500, OmniMath (500), AIME, MBPP+ (500), HumanEval+ (500), LiveCodeBench (500)

Think prefix logic:
- All base/instruct models: think_prefix=False on all datasets

Usage:
  python eval_base_models.py                         # all 4 base models
  python eval_base_models.py --base-model llama8b    # single model
  python eval_base_models.py --max-tokens 8192       # custom max_tokens
  python eval_base_models.py --max-problems 50       # quick check
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
    BASE_MODELS, DEFAULT_MAX_LENGTH, RESULTS_BASE,
    build_base_eval_name,
)


EVAL_SCRIPT = Path(__file__).parent.parent / "evaluate.py"

# Full eval suite
EVAL_DATASETS = "math_500,omni_math,aime,mbppplus,humanevalplus,livecodebench_v5"

# All datasets have think_prefix disabled for base models
ALL_DATASETS_NO_THINK = "math_500,omni_math,aime,mbppplus,humanevalplus,livecodebench_v5"


def eval_base_model(base_short: str, base_model: str,
                    max_tokens: int, max_problems: int | None,
                    datasets: str = EVAL_DATASETS) -> bool:
    """Evaluate a single base model. Returns True on success."""
    results_name = build_base_eval_name(base_short, max_tokens)

    print(f"\n{'='*60}")
    print(f"Evaluating base model: {base_short} ({base_model})")
    print(f"  Results: {RESULTS_BASE / results_name}")
    print(f"  Datasets: {datasets}")
    print(f"  Max tokens: {max_tokens}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--base-model-name", base_model,
        "--results-name", results_name,
        "--results-base", str(RESULTS_BASE),
        "--datasets", datasets,
        "--max-tokens", str(max_tokens),
        "--no-think-prefix-datasets", ALL_DATASETS_NO_THINK,
    ]
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
    print(f"\n{base_short}: {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate base models")
    parser.add_argument("--base-model", type=str, default=None,
                        choices=list(BASE_MODELS.keys()),
                        help="Evaluate only this base model")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max tokens for generation (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit problems per dataset")
    parser.add_argument("--datasets", type=str, default=EVAL_DATASETS,
                        help="Comma-separated dataset list")
    args = parser.parse_args()

    models = BASE_MODELS
    if args.base_model:
        models = {args.base_model: BASE_MODELS[args.base_model]}

    print(f"Base model evaluation (25_02)")
    print(f"  Models: {', '.join(models.keys())}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Max tokens: {args.max_tokens}")

    t0 = time.time()
    results = {}
    for short, full in models.items():
        ok = eval_base_model(short, full, args.max_tokens, args.max_problems, args.datasets)
        results[short] = ok

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"BASE MODEL EVALUATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    for short, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {short}: {status}")


if __name__ == "__main__":
    main()

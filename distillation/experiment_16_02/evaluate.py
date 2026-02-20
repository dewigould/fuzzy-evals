"""Experiment 16/02 — Evaluation sweep.

Evaluate a base model or checkpoint on MATH-500, AIME, HumanEval+, MBPP+.

Usage:
  # Evaluate raw base model (full sweep)
  python evaluate.py base --results-name qwen3_base

  # Evaluate a trained checkpoint
  python evaluate.py checkpoint --sampler-path "tinker://..." --results-name math_step100

  # Fast mode: math_500 (100) + mbppplus (100) — ~5 min per checkpoint
  python evaluate.py checkpoint --sampler-path "tinker://..." --results-name math_step100 --fast

  # Specific datasets only
  python evaluate.py base --results-name test --datasets math_500,aime
"""

import argparse
import asyncio
import importlib
import json
import sys
import time
from pathlib import Path

# Add parent distillation/ to path for shared imports
EXPERIMENT_DIR = Path(__file__).parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, "/workspace/tinker-cookbook")

from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT  # noqa: E402 — system prompts only
from infer import setup_renderer_and_tokenizer, create_base_client, create_checkpoint_client  # noqa: E402

from experiment_16_02.config import (  # noqa: E402
    MODEL_NAME, INFERENCE_MAX_TOKENS, EVAL_DATASETS, EVAL_CONCURRENCY, RESULTS_DIR,
)

# Dataset name → eval_tools module mapping (same as parent evaluate.py)
DATASET_MODULES = {
    "math_500": "eval_tools.math_500",
    "aime": "eval_tools.aime",
    "humanevalplus": "eval_tools.humanevalplus",
    "mbppplus": "eval_tools.mbppplus",
    "kodcode_500": "eval_tools.kodcode",
    "codeforces_500": "eval_tools.codeforces",
    "livecodebench_v5": "eval_tools.livecodebench",
    "omni_math": "eval_tools.omni_math",
    "fuzzy_philosophy": "eval_tools.fuzzy_philosophy",
    "fuzzy_weird_qs": "eval_tools.fuzzy_weird_qs",
    "fuzzy_futuristic_tech": "eval_tools.fuzzy_futuristic_tech",
}


async def evaluate_model(
    model_type: str,
    sampler_path: str | None,
    results_name: str,
    datasets: list[str],
    max_tokens: int,
    max_problems: int | None,
    model_name: str,
    temperature: float = 0.0,
    top_p: float | None = None,
):
    """Run eval sweep on the specified datasets."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {results_name}")
    print(f"  Type: {model_type}, Model: {model_name}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max tokens: {max_tokens}")
    if max_problems is not None:
        print(f"  Max problems: {max_problems}")
    print(f"{'='*60}")

    # Set up renderer and tokenizer
    renderer, tokenizer = setup_renderer_and_tokenizer(model_name)

    # Create sampling client
    if model_type == "base":
        print(f"  Loading base model: {model_name}")
        sampling_client = create_base_client(model_name)
        think_prefix = False
    elif model_type == "checkpoint":
        if not sampler_path:
            print("ERROR: --sampler-path required for checkpoint mode")
            sys.exit(1)
        print(f"  Loading checkpoint: {sampler_path}")
        sampling_client = create_checkpoint_client(sampler_path)
        think_prefix = True
    else:
        print(f"ERROR: Unknown model type '{model_type}'")
        sys.exit(1)

    # Set up results directory
    results_dir = RESULTS_DIR / results_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate datasets
    valid_datasets = [ds for ds in datasets if ds in DATASET_MODULES]
    for ds in datasets:
        if ds not in DATASET_MODULES:
            print(f"  WARNING: Unknown dataset '{ds}', skipping")

    # Run all datasets concurrently
    async def run_dataset(ds_name):
        print(f"\n  --- {ds_name} (think_prefix={think_prefix}, max_tokens={max_tokens}) ---")
        t0 = time.time()
        module = importlib.import_module(DATASET_MODULES[ds_name])
        extra_kwargs = {}
        if max_problems is not None:
            extra_kwargs["max_problems"] = max_problems
        if temperature != 0.0:
            extra_kwargs["temperature"] = temperature
        if top_p is not None:
            extra_kwargs["top_p"] = top_p
        result = await module.run(
            sampling_client, renderer, tokenizer, results_dir, results_name,
            think_prefix=think_prefix, max_tokens=max_tokens,
            **extra_kwargs,
        )
        elapsed = time.time() - t0
        print(f"  {ds_name} completed in {elapsed:.0f}s")
        return ds_name, result

    dataset_results = await asyncio.gather(*[run_dataset(ds) for ds in valid_datasets])
    all_results = dict(dataset_results)

    # Save summary
    summary_path = results_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved eval summary to {summary_path}")

    # Print results table
    print(f"\n{'='*60}")
    print(f"RESULTS: {results_name}")
    print(f"{'='*60}")
    for ds_name, ds_result in all_results.items():
        if "accuracy" in ds_result:
            print(f"  {ds_name:<20} {ds_result['accuracy']*100:.1f}% ({ds_result['n_correct']}/{ds_result['total']})")
        elif "mean_score" in ds_result:
            print(f"  {ds_name:<20} {ds_result['mean_score']:.2f}")

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Experiment 16/02: Evaluation sweep")
    parser.add_argument("mode", choices=["base", "checkpoint"],
                        help="'base' for raw model, 'checkpoint' for trained checkpoint")
    parser.add_argument("--sampler-path", type=str, default=None,
                        help="tinker:// sampler path (required for checkpoint mode)")
    parser.add_argument("--results-name", type=str, required=True,
                        help="Name for results directory")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: math_500 (100) + mbppplus (100), ~5 min per checkpoint")
    parser.add_argument("--datasets", type=str, default=None,
                        help=f"Comma-separated datasets (default: {','.join(EVAL_DATASETS)})")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit problems per dataset (default: full dataset)")
    parser.add_argument("--max-tokens", type=int, default=INFERENCE_MAX_TOKENS,
                        help=f"Max tokens for generation (default: {INFERENCE_MAX_TOKENS})")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help=f"Base model name (default: {MODEL_NAME})")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-p / nucleus sampling (default: None)")
    args = parser.parse_args()

    if args.fast:
        datasets = ["math_500", "mbppplus"]
        max_problems = args.max_problems if args.max_problems is not None else 100
    else:
        datasets = args.datasets.split(",") if args.datasets else EVAL_DATASETS
        max_problems = args.max_problems

    await evaluate_model(
        model_type=args.mode,
        sampler_path=args.sampler_path,
        results_name=args.results_name,
        datasets=datasets,
        max_tokens=args.max_tokens,
        max_problems=max_problems,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    asyncio.run(main())

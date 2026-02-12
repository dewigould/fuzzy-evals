"""Central evaluation script for the distillation experiment.

Reads eval_config.yaml, evaluates specified models on specified datasets,
and saves results as parquet files to results/{model_name}/.

Usage:
  python evaluate.py                                    # Run all from config
  python evaluate.py --model base                       # Evaluate base model only
  python evaluate.py --model base --datasets math_500,aime  # Specific datasets
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

from config import MODEL_NAME, RESULTS_BASE
from infer import setup_renderer_and_tokenizer, create_base_client, create_checkpoint_client

# Dataset name → eval_tools module mapping
DATASET_MODULES = {
    "math_500": "eval_tools.math_500",
    "aime": "eval_tools.aime",
    "kodcode_500": "eval_tools.kodcode",
    "codeforces_500": "eval_tools.codeforces",
    "fuzzy_philosophy": "eval_tools.fuzzy_philosophy",
    "fuzzy_weird_qs": "eval_tools.fuzzy_weird_qs",
    "fuzzy_futuristic_tech": "eval_tools.fuzzy_futuristic_tech",
}


def load_config(config_path: str = "eval_config.yaml") -> dict:
    """Load evaluation config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_best_checkpoint(task: str, config_name: str) -> str | None:
    """Find the best checkpoint sampler_path from training results."""
    from config import TRAINING_BASE
    task_dir = Path(TRAINING_BASE) / task

    # Check sweep summary first
    summary_path = task_dir / "sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        winner = summary.get("winner", config_name)
        winner_dir = task_dir / winner
    else:
        winner_dir = task_dir / config_name

    # Find best checkpoint by lowest test NLL
    metrics_file = winner_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    best_step = None
    best_nll = float("inf")
    with open(metrics_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "test/nll" in d:
                nll = d["test/nll"]
                step = d.get("step", d.get("progress/batch", -1))
                if nll < best_nll:
                    best_nll = nll
                    best_step = step

    if best_step is None:
        return None

    ckpt_file = winner_dir / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None

    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Find exact or closest checkpoint
    best_ckpt = None
    for ckpt in checkpoints:
        if ckpt.get("batch", -1) == best_step and "sampler_path" in ckpt:
            best_ckpt = ckpt
            break

    if best_ckpt is None:
        valid = [c for c in checkpoints if "sampler_path" in c]
        if valid:
            best_ckpt = min(valid, key=lambda c: abs(c.get("batch", 0) - best_step))

    if best_ckpt is None:
        return None

    print(f"  Found checkpoint: step {best_step}, NLL {best_nll:.4f}, path: {best_ckpt['sampler_path']}")
    return best_ckpt["sampler_path"]


async def evaluate_model(model_name: str, model_config: dict, datasets: list[str],
                         default_max_tokens: int = 8192,
                         max_problems: int | None = None,
                         fuzzy_samples: int | None = None):
    """Evaluate a single model on specified datasets."""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"{'='*60}")

    # Set up renderer and tokenizer
    renderer, tokenizer = setup_renderer_and_tokenizer()

    # Create sampling client
    model_type = model_config.get("type", "base_model")
    if model_type == "base_model":
        base_name = model_config.get("name", MODEL_NAME)
        print(f"  Loading base model: {base_name}")
        sampling_client = create_base_client(base_name)
    elif model_type == "checkpoint":
        sampler_path = model_config.get("sampler_path")
        if sampler_path is None:
            # Try auto-finding from training results
            if "math" in model_name:
                sampler_path = find_best_checkpoint("math", "")
            elif "code" in model_name:
                sampler_path = find_best_checkpoint("code", "")

        if sampler_path is None:
            print(f"  ERROR: No sampler_path for {model_name}. Skipping.")
            return {}
        print(f"  Loading checkpoint: {sampler_path}")
        sampling_client = create_checkpoint_client(sampler_path)
    else:
        print(f"  ERROR: Unknown model type '{model_type}'. Skipping.")
        return {}

    # Set up results directory
    results_dir = RESULTS_BASE / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-model inference overrides
    no_think_datasets = set(model_config.get("no_think_prefix_datasets", []))
    model_max_tokens = model_config.get("max_tokens", default_max_tokens)

    # Run each dataset evaluation
    all_results = {}
    for ds_name in datasets:
        if ds_name not in DATASET_MODULES:
            print(f"  WARNING: Unknown dataset '{ds_name}'. Skipping.")
            continue

        # Determine per-dataset inference settings
        use_think_prefix = ds_name not in no_think_datasets
        print(f"\n  --- {ds_name} (think_prefix={use_think_prefix}, max_tokens={model_max_tokens}) ---")
        t0 = time.time()

        # Import and run the eval module
        import importlib
        module = importlib.import_module(DATASET_MODULES[ds_name])
        extra_kwargs = {}
        if max_problems is not None:
            extra_kwargs["max_problems"] = max_problems
        if fuzzy_samples is not None:
            extra_kwargs["fuzzy_samples"] = fuzzy_samples
        result = await module.run(
            sampling_client, renderer, tokenizer, results_dir, model_name,
            think_prefix=use_think_prefix, max_tokens=model_max_tokens,
            **extra_kwargs,
        )

        elapsed = time.time() - t0
        print(f"  {ds_name} completed in {elapsed:.0f}s")
        all_results[ds_name] = result

    # Save summary
    summary_path = results_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved eval summary to {summary_path}")

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Run distillation evaluations")
    parser.add_argument("--config", default="eval_config.yaml",
                        help="Path to eval config YAML")
    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate only this model (default: all)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of datasets (default: all from config)")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems per dataset (for quick sanity checks)")
    parser.add_argument("--fuzzy-samples", type=int, default=None,
                        help="Override number of samples per fuzzy question (default: 10)")
    args = parser.parse_args()

    config = load_config(args.config)
    models = config.get("models", {})
    default_datasets = config.get("datasets", list(DATASET_MODULES.keys()))

    # Filter to requested model(s)
    if args.model:
        if args.model not in models:
            print(f"ERROR: Model '{args.model}' not found in config")
            print(f"Available models: {', '.join(models.keys())}")
            sys.exit(1)
        models = {args.model: models[args.model]}

    # Filter datasets
    datasets = args.datasets.split(",") if args.datasets else default_datasets

    print(f"Distillation Evaluation")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Datasets: {', '.join(datasets)}")

    t0 = time.time()
    all_model_results = {}

    inference_config = config.get("inference", {})
    default_max_tokens = inference_config.get("max_tokens", 8192)

    for model_name, model_config in models.items():
        result = await evaluate_model(model_name, model_config, datasets, default_max_tokens,
                                       max_problems=args.max_problems,
                                       fuzzy_samples=args.fuzzy_samples)
        all_model_results[model_name] = result

    elapsed = time.time() - t0

    # Print final summary table
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (total time: {elapsed:.0f}s)")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Dataset':<20} {'Metric':<15} {'Value':>10}")
    print(f"{'-'*70}")

    for model_name, model_results in all_model_results.items():
        for ds_name, ds_result in model_results.items():
            if "accuracy" in ds_result:
                metric = "accuracy"
                value = f"{ds_result['accuracy']*100:.1f}%"
            elif "mean_score" in ds_result:
                metric = "mean_score"
                value = f"{ds_result['mean_score']:.2f}"
            else:
                metric = "n/a"
                value = "n/a"
            print(f"{model_name:<25} {ds_name:<20} {metric:<15} {value:>10}")

    print(f"\nAll results saved to {RESULTS_BASE}/")


if __name__ == "__main__":
    asyncio.run(main())

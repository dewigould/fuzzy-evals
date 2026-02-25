"""Evaluate ALL checkpoints from thinking-trace training runs.

Stage 1: Evaluate every checkpoint on math_500 and mbpp+ (100 samples).
  - Best math checkpoint = highest math_500 accuracy
  - Best code checkpoint = highest mbpp+ accuracy

Stage 2: Evaluate the winning checkpoints on aime and humaneval+.

Usage:
  python eval_thinking_checkpoints.py
  python eval_thinking_checkpoints.py --run thinking_math_base   # Single run
  python eval_thinking_checkpoints.py --max-problems 50          # Override
  python eval_thinking_checkpoints.py --stage1-only              # Skip stage 2
"""

import argparse
import asyncio
import importlib
import json
import sys
import time
from pathlib import Path

from config import TRAINING_BASE, RESULTS_BASE
from infer import setup_renderer_and_tokenizer, create_checkpoint_client

# The 4 thinking training runs
THINKING_RUNS = {
    "thinking_math_base": {
        "base_model": "Qwen/Qwen3-30B-A3B-Base",
        "no_think_datasets": [],  # base model: always use think_prefix
    },
    "thinking_math_it": {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "no_think_datasets": ["math_500"],  # IT model generates <think> natively on math
    },
    "thinking_code_base": {
        "base_model": "Qwen/Qwen3-30B-A3B-Base",
        "no_think_datasets": [],
    },
    "thinking_code_it": {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "no_think_datasets": ["mbppplus"],  # IT model generates <think> natively on code
    },
}

# Only these two benchmarks — used to pick best math and best code checkpoints
EVAL_DATASETS = ["math_500", "mbppplus"]
DEFAULT_MAX_PROBLEMS = 100

DATASET_MODULES = {
    "math_500": "eval_tools.math_500",
    "mbppplus": "eval_tools.mbppplus",
    "aime": "eval_tools.aime",
    "humanevalplus": "eval_tools.humanevalplus",
}

# Stage 1: sweep all checkpoints on these
STAGE1_DATASETS = ["math_500", "mbppplus"]
# Stage 2: run winners on these
STAGE2_DATASETS = ["aime", "humanevalplus"]


def find_all_checkpoints(run_name: str) -> list[dict]:
    """Find all checkpoints for a run, sorted by step."""
    run_dir = Path(TRAINING_BASE) / run_name / "A_r32_high_lr"
    ckpt_file = run_dir / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []

    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                ckpt = json.loads(line)
                if "sampler_path" in ckpt:
                    checkpoints.append(ckpt)
            except json.JSONDecodeError:
                continue

    return sorted(checkpoints, key=lambda c: c.get("batch", 0))


async def evaluate_one_checkpoint(
    sampling_client, renderer, tokenizer,
    run_name: str, step: int, model_name: str,
    no_think_datasets: set[str],
    max_problems: int,
    datasets: list[str] | None = None,
) -> dict:
    """Evaluate a single checkpoint on specified datasets."""
    if datasets is None:
        datasets = STAGE1_DATASETS
    results_name = f"{run_name}_step{step}"
    results_dir = RESULTS_BASE / results_name
    results_dir.mkdir(parents=True, exist_ok=True)

    ds_results = {}
    for ds_name in datasets:
        use_think_prefix = ds_name not in no_think_datasets
        print(f"    [{ds_name}] think_prefix={use_think_prefix}")
        t0 = time.time()
        module = importlib.import_module(DATASET_MODULES[ds_name])
        result = await module.run(
            sampling_client, renderer, tokenizer,
            results_dir, results_name,
            think_prefix=use_think_prefix,
            max_tokens=8192,
            max_problems=max_problems,
        )
        elapsed = time.time() - t0
        ds_results[ds_name] = result
        acc = result.get("accuracy", 0)
        print(f"    [{ds_name}] {acc*100:.1f}% ({elapsed:.0f}s)")

    # Save per-checkpoint summary
    with open(results_dir / "eval_summary.json", "w") as f:
        json.dump(ds_results, f, indent=2)

    return ds_results


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all thinking-trace checkpoints on math_500 and mbpp+"
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Evaluate only this run (e.g. thinking_math_base)")
    parser.add_argument("--max-problems", type=int, default=DEFAULT_MAX_PROBLEMS,
                        help="Max problems per dataset (default: 100)")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Only run stage 1 (checkpoint sweep), skip stage 2 (winner evals)")
    args = parser.parse_args()

    runs = THINKING_RUNS
    if args.run:
        if args.run not in THINKING_RUNS:
            print(f"Unknown run: {args.run}")
            print(f"Available: {', '.join(THINKING_RUNS.keys())}")
            sys.exit(1)
        runs = {args.run: THINKING_RUNS[args.run]}

    print(f"Evaluating ALL checkpoints from thinking-trace training runs")
    print(f"Runs: {', '.join(runs.keys())}")
    print(f"Datasets: {', '.join(EVAL_DATASETS)}")
    print(f"Max problems per dataset: {args.max_problems}")
    print()

    # Collect all results: {run_name: [{step, sampler_path, math_500_acc, mbppplus_acc}, ...]}
    all_run_results = {}
    t0_global = time.time()

    for run_name, run_config in runs.items():
        print(f"\n{'='*70}")
        print(f"Run: {run_name} (model: {run_config['base_model']})")
        print(f"{'='*70}")

        checkpoints = find_all_checkpoints(run_name)
        if not checkpoints:
            print(f"  No checkpoints found. Skipping.")
            continue

        print(f"  Found {len(checkpoints)} checkpoints")

        # Set up renderer and client once, reuse across checkpoints
        renderer, tokenizer = setup_renderer_and_tokenizer(run_config["base_model"])
        no_think = set(run_config["no_think_datasets"])

        checkpoint_results = []
        for ckpt in checkpoints:
            step = ckpt.get("batch", 0)
            sampler_path = ckpt["sampler_path"]
            print(f"\n  --- Step {step} ---")

            sampling_client = create_checkpoint_client(sampler_path)

            ds_results = await evaluate_one_checkpoint(
                sampling_client, renderer, tokenizer,
                run_name, step, run_name,
                no_think, args.max_problems,
            )

            checkpoint_results.append({
                "step": step,
                "sampler_path": sampler_path,
                "math_500_acc": ds_results.get("math_500", {}).get("accuracy", 0),
                "mbppplus_acc": ds_results.get("mbppplus", {}).get("accuracy", 0),
                "full_results": ds_results,
            })

        all_run_results[run_name] = checkpoint_results

    elapsed_global = time.time() - t0_global

    # ── Print per-run checkpoint table ──────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"CHECKPOINT SWEEP RESULTS (total time: {elapsed_global:.0f}s)")
    print(f"{'='*80}")

    # Track best per model type
    best_by_model = {}  # e.g. {"base": {"best_math": ..., "best_code": ...}, "it": ...}

    for run_name, checkpoint_results in all_run_results.items():
        if not checkpoint_results:
            continue

        # Determine model type from run name
        model_type = "base" if "base" in run_name else "it"
        if model_type not in best_by_model:
            best_by_model[model_type] = {"best_math": None, "best_code": None}

        print(f"\n  {run_name}:")
        print(f"  {'Step':>6}  {'math_500':>10}  {'mbpp+':>10}")
        print(f"  {'-'*30}")

        for cr in checkpoint_results:
            math_str = f"{cr['math_500_acc']*100:.1f}%"
            code_str = f"{cr['mbppplus_acc']*100:.1f}%"
            print(f"  {cr['step']:>6}  {math_str:>10}  {code_str:>10}")

        # Find best math checkpoint for this run
        best_math = max(checkpoint_results, key=lambda x: x["math_500_acc"])
        # Find best code checkpoint for this run
        best_code = max(checkpoint_results, key=lambda x: x["mbppplus_acc"])

        print(f"\n  Best math (math_500): step {best_math['step']} "
              f"({best_math['math_500_acc']*100:.1f}%)")
        print(f"  Best code (mbpp+):    step {best_code['step']} "
              f"({best_code['mbppplus_acc']*100:.1f}%)")

        # Track across runs for each model type
        cur_best_math = best_by_model[model_type]["best_math"]
        if cur_best_math is None or best_math["math_500_acc"] > cur_best_math["math_500_acc"]:
            best_by_model[model_type]["best_math"] = {
                **best_math, "run": run_name,
            }

        cur_best_code = best_by_model[model_type]["best_code"]
        if cur_best_code is None or best_code["mbppplus_acc"] > cur_best_code["mbppplus_acc"]:
            best_by_model[model_type]["best_code"] = {
                **best_code, "run": run_name,
            }

    # ── Print winners ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"WINNERS")
    print(f"{'='*80}")

    for model_type in ["base", "it"]:
        if model_type not in best_by_model:
            continue
        bests = best_by_model[model_type]
        model_label = "Qwen3-Base" if model_type == "base" else "Qwen3-Instruct"
        print(f"\n  {model_label}:")

        if bests["best_math"]:
            bm = bests["best_math"]
            print(f"    Best math distill: {bm['run']} step {bm['step']} "
                  f"— math_500={bm['math_500_acc']*100:.1f}% "
                  f"mbpp+={bm['mbppplus_acc']*100:.1f}%")
            print(f"      sampler_path: {bm['sampler_path']}")

        if bests["best_code"]:
            bc = bests["best_code"]
            print(f"    Best code distill: {bc['run']} step {bc['step']} "
                  f"— math_500={bc['math_500_acc']*100:.1f}% "
                  f"mbpp+={bc['mbppplus_acc']*100:.1f}%")
            print(f"      sampler_path: {bc['sampler_path']}")

    # ── Stage 2: Evaluate winners on aime + humaneval+ ─────────────────────
    stage2_results = {}
    if not args.stage1_only:
        print(f"\n{'='*80}")
        print(f"STAGE 2: Evaluating winners on {', '.join(STAGE2_DATASETS)}")
        print(f"{'='*80}")

        # Collect unique winner checkpoints to evaluate
        winners_to_eval = []  # (label, sampler_path, base_model, no_think_datasets)
        for model_type in ["base", "it"]:
            if model_type not in best_by_model:
                continue
            bests = best_by_model[model_type]
            base_model = ("Qwen/Qwen3-30B-A3B-Base" if model_type == "base"
                          else "Qwen/Qwen3-30B-A3B-Instruct-2507")
            # no_think for stage 2: IT model natively generates <think> on in-domain
            no_think = set()
            if model_type == "it":
                no_think = {"aime", "humanevalplus"}

            if bests["best_math"]:
                bm = bests["best_math"]
                label = f"winner_{model_type}_math_step{bm['step']}"
                winners_to_eval.append((label, bm["sampler_path"], base_model, no_think))

            if bests["best_code"]:
                bc = bests["best_code"]
                label = f"winner_{model_type}_code_step{bc['step']}"
                # Avoid duplicating if same checkpoint won both
                if not any(w[1] == bc["sampler_path"] for w in winners_to_eval):
                    winners_to_eval.append((label, bc["sampler_path"], base_model, no_think))

        for label, sampler_path, base_model, no_think in winners_to_eval:
            print(f"\n  --- {label} ---")
            renderer, tokenizer = setup_renderer_and_tokenizer(base_model)
            sampling_client = create_checkpoint_client(sampler_path)

            ds_results = await evaluate_one_checkpoint(
                sampling_client, renderer, tokenizer,
                label, 0, label, no_think, args.max_problems,
                datasets=STAGE2_DATASETS,
            )
            stage2_results[label] = ds_results

            for ds_name, res in ds_results.items():
                acc = res.get("accuracy", 0)
                print(f"    {ds_name}: {acc*100:.1f}%")

        print(f"\n{'='*80}")
        print(f"STAGE 2 RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<40} {'Dataset':<20} {'Accuracy':>10}")
        print(f"{'-'*70}")
        for label, ds_results in stage2_results.items():
            for ds_name, res in ds_results.items():
                acc = res.get("accuracy", 0)
                print(f"{label:<40} {ds_name:<20} {acc*100:.1f}%")

    # ── Save full results ───────────────────────────────────────────────────
    summary = {
        "stage1_all_checkpoints": {
            run_name: [
                {k: v for k, v in cr.items() if k != "full_results"}
                for cr in crs
            ]
            for run_name, crs in all_run_results.items()
        },
        "winners": {
            model_type: {
                "best_math": {k: v for k, v in bests["best_math"].items() if k != "full_results"} if bests["best_math"] else None,
                "best_code": {k: v for k, v in bests["best_code"].items() if k != "full_results"} if bests["best_code"] else None,
            }
            for model_type, bests in best_by_model.items()
        },
        "stage2_winner_evals": {
            label: {ds: {k: v for k, v in res.items()} for ds, res in ds_results.items()}
            for label, ds_results in stage2_results.items()
        },
    }

    summary_path = RESULTS_BASE / "thinking_eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full results to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())

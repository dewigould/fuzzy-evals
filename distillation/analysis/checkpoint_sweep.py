"""Checkpoint sweep: evaluate multiple training checkpoints on AIME and KodCode.

Sweeps through checkpoints at (0, 5K, 10K, 15K, 20K, 35K, 50K) examples
for both math-distilled and code-distilled models.

Maximally parallelized: all (checkpoint × dataset) pairs run concurrently.
That's 7 checkpoints × 2 datasets × 2 model families = 26 eval jobs at once
(base model shared, so 2 + 6×4 = 26).
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAME, RESULTS_BASE
from infer import setup_renderer_and_tokenizer, create_base_client, create_checkpoint_client

MATH_CKPT_BASE = "tinker://20cab21e-fd37-5d46-92b3-3a9e0855bb90:train:0/sampler_weights"
CODE_CKPT_BASE = "tinker://f136a2a0-f167-53f9-9e81-db0a13641117:train:0/sampler_weights"

# (examples_seen, step_name)
SWEEP_POINTS = [
    (0, None),       # base model
    (5_000, "000100"),
    (10_000, "000200"),
    (15_000, "000300"),
    (20_000, "000400"),
    (35_000, "000700"),
    (50_000, "001000"),
]

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_RESULTS_DIR = RESULTS_BASE / "checkpoint_sweep"
SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_MODULES = {
    "aime": "eval_tools.aime",
    "kodcode_500": "eval_tools.kodcode",
}


async def eval_one(sampling_client, renderer, tokenizer,
                   dataset_name: str, model_label: str,
                   think_prefix: bool, max_tokens: int,
                   max_problems: int) -> dict:
    """Run a single dataset eval and return the result dict."""
    import importlib

    results_dir = SWEEP_RESULTS_DIR / model_label
    results_dir.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(DATASET_MODULES[dataset_name])
    result = await module.run(
        sampling_client, renderer, tokenizer, results_dir, model_label,
        think_prefix=think_prefix, max_tokens=max_tokens,
        max_problems=max_problems,
    )
    return result


async def main():
    max_tokens_ckpt = 28672
    max_tokens_base = 8192
    max_problems = 100  # AIME has 90, KodCode capped at 100

    renderer, tokenizer = setup_renderer_and_tokenizer()
    t0 = time.time()

    # Build all jobs upfront: list of (key, coro) where key identifies the result slot
    # key = (model_family, dataset, examples)
    jobs = []

    # Create all clients first
    clients = {}
    print("Creating sampling clients...")

    base_client = create_base_client()
    clients["base"] = base_client

    for examples, step_name in SWEEP_POINTS:
        if step_name is None:
            continue
        math_path = f"{MATH_CKPT_BASE}/{step_name}"
        code_path = f"{CODE_CKPT_BASE}/{step_name}"
        print(f"  Loading math step {step_name} and code step {step_name}...")
        clients[f"math_{step_name}"] = create_checkpoint_client(math_path)
        clients[f"code_{step_name}"] = create_checkpoint_client(code_path)

    print(f"All {len(clients)} clients created in {time.time()-t0:.0f}s")
    print(f"\nLaunching all eval jobs in parallel...")

    # Base model jobs (shared across both model families at x=0)
    jobs.append((
        ("base", "aime", 0),
        eval_one(base_client, renderer, tokenizer, "aime", "base_0",
                 think_prefix=False, max_tokens=max_tokens_base, max_problems=max_problems)
    ))
    jobs.append((
        ("base", "kodcode", 0),
        eval_one(base_client, renderer, tokenizer, "kodcode_500", "base_0",
                 think_prefix=False, max_tokens=max_tokens_base, max_problems=max_problems)
    ))

    # Checkpoint jobs
    for examples, step_name in SWEEP_POINTS:
        if step_name is None:
            continue

        math_client = clients[f"math_{step_name}"]
        code_client = clients[f"code_{step_name}"]
        math_label = f"math_step{step_name}"
        code_label = f"code_step{step_name}"

        # Math checkpoint on AIME (in-domain, no think prefix)
        jobs.append((
            ("math", "aime", examples),
            eval_one(math_client, renderer, tokenizer, "aime", math_label,
                     think_prefix=False, max_tokens=max_tokens_ckpt, max_problems=max_problems)
        ))
        # Math checkpoint on KodCode (cross-domain, think prefix)
        jobs.append((
            ("math", "kodcode", examples),
            eval_one(math_client, renderer, tokenizer, "kodcode_500", math_label,
                     think_prefix=True, max_tokens=max_tokens_ckpt, max_problems=max_problems)
        ))
        # Code checkpoint on AIME (cross-domain, think prefix)
        jobs.append((
            ("code", "aime", examples),
            eval_one(code_client, renderer, tokenizer, "aime", code_label,
                     think_prefix=True, max_tokens=max_tokens_ckpt, max_problems=max_problems)
        ))
        # Code checkpoint on KodCode (in-domain, no think prefix)
        jobs.append((
            ("code", "kodcode", examples),
            eval_one(code_client, renderer, tokenizer, "kodcode_500", code_label,
                     think_prefix=False, max_tokens=max_tokens_ckpt, max_problems=max_problems)
        ))

    print(f"Total jobs: {len(jobs)}")

    # Run ALL jobs concurrently
    keys = [k for k, _ in jobs]
    coros = [c for _, c in jobs]
    results_list = await asyncio.gather(*coros)

    elapsed = time.time() - t0
    print(f"\nAll {len(jobs)} jobs completed in {elapsed:.0f}s")

    # Organize results
    all_results = {
        "math": {"aime": [], "kodcode": []},
        "code": {"aime": [], "kodcode": []},
    }

    # First insert base results (shared for both families at x=0)
    base_aime = None
    base_kodcode = None
    for key, result in zip(keys, results_list):
        if key == ("base", "aime", 0):
            base_aime = result
        elif key == ("base", "kodcode", 0):
            base_kodcode = result

    for family in ["math", "code"]:
        all_results[family]["aime"].append((0, base_aime))
        all_results[family]["kodcode"].append((0, base_kodcode))

    # Insert checkpoint results
    for key, result in zip(keys, results_list):
        family, ds, examples = key
        if family == "base":
            continue
        all_results[family][ds].append((examples, result))

    # Sort by examples
    for family in all_results:
        for ds in all_results[family]:
            all_results[family][ds].sort(key=lambda x: x[0])

    # Save raw results
    serializable = {}
    for model_type in all_results:
        serializable[model_type] = {}
        for ds in all_results[model_type]:
            serializable[model_type][ds] = [
                {"examples": ex, "result": r} for ex, r in all_results[model_type][ds]
            ]
    with open(SWEEP_RESULTS_DIR / "sweep_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved sweep results to {SWEEP_RESULTS_DIR / 'sweep_results.json'}")

    # Print summary table
    print(f"\n{'='*80}")
    print("CHECKPOINT SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Examples':>10} | {'Math→AIME':>12} {'Math→KodCode':>14} | {'Code→AIME':>12} {'Code→KodCode':>14}")
    print("-" * 80)
    for i, (examples, _) in enumerate(all_results["math"]["aime"]):
        math_aime = all_results["math"]["aime"][i][1]["accuracy"] * 100
        math_kc = all_results["math"]["kodcode"][i][1]["accuracy"] * 100
        code_aime = all_results["code"]["aime"][i][1]["accuracy"] * 100
        code_kc = all_results["code"]["kodcode"][i][1]["accuracy"] * 100
        print(f"{examples:>10,} | {math_aime:>11.1f}% {math_kc:>13.1f}% | {code_aime:>11.1f}% {code_kc:>13.1f}%")

    # Generate plots
    generate_plots(all_results)


def generate_plots(all_results):
    """Generate accuracy vs training examples plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"math": "#fd8d3c", "code": "#74c476"}
    labels = {"math": "Math Distill", "code": "Code Distill"}
    markers = {"math": "o", "code": "s"}

    # Plot 1: AIME accuracy
    ax = axes[0]
    for model_type in ["math", "code"]:
        points = all_results[model_type]["aime"]
        xs = [p[0] / 1000 for p in points]
        ys = [p[1]["accuracy"] * 100 for p in points]
        ax.plot(xs, ys, f"{markers[model_type]}-", color=colors[model_type],
                label=labels[model_type], linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=8)

    ax.axhline(y=all_results["math"]["aime"][0][1]["accuracy"] * 100,
               color="gray", linestyle="--", alpha=0.5, label="Base model")
    ax.set_xlabel("Training Examples (thousands)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("AIME Accuracy vs Training Examples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 2: KodCode accuracy
    ax = axes[1]
    for model_type in ["math", "code"]:
        points = all_results[model_type]["kodcode"]
        xs = [p[0] / 1000 for p in points]
        ys = [p[1]["accuracy"] * 100 for p in points]
        ax.plot(xs, ys, f"{markers[model_type]}-", color=colors[model_type],
                label=labels[model_type], linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=8)

    ax.axhline(y=all_results["math"]["kodcode"][0][1]["accuracy"] * 100,
               color="gray", linestyle="--", alpha=0.5, label="Base model")
    ax.set_xlabel("Training Examples (thousands)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("KodCode Accuracy vs Training Examples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Distillation Checkpoint Sweep: Accuracy vs Training Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOT_DIR / "checkpoint_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.close(fig)


if __name__ == "__main__":
    asyncio.run(main())

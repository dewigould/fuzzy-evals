"""Evaluation module. Runs eval datasets on base models or checkpoints.

Wraps eval_tools/ modules. Handles think-prefix logic and OOM-safe
concurrency for code eval datasets.

Usage:
    from evaluate import evaluate_checkpoint, evaluate_base_model

    results = await evaluate_checkpoint(
        sampler_path="tinker://...",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        datasets=["math_500", "mbppplus"],
        results_dir=Path("logs/my_exp/checkpoint_eval/step100"),
        task="math",
    )
"""

import asyncio
import importlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/workspace/tinker-cookbook")

from config import CODE_DATASETS, MATH_DATASETS, MAX_CONCURRENT_CODE_EVALS, MAX_CONCURRENT_MATH_EVALS
from infer import setup_renderer_and_tokenizer, create_base_client, create_checkpoint_client

# Dataset name → eval_tools module mapping
DATASET_MODULES = {
    "math_500": "eval_tools.math_500",
    "aime": "eval_tools.aime",
    "omni_math": "eval_tools.omni_math",
    "kodcode_500": "eval_tools.kodcode",
    "codeforces_500": "eval_tools.codeforces",
    "livecodebench_v5": "eval_tools.livecodebench",
    "humanevalplus": "eval_tools.humanevalplus",
    "mbppplus": "eval_tools.mbppplus",
    "fuzzy_philosophy": "eval_tools.fuzzy_philosophy",
    "fuzzy_weird_qs": "eval_tools.fuzzy_weird_qs",
    "fuzzy_futuristic_tech": "eval_tools.fuzzy_futuristic_tech",
}

# Semaphores limiting concurrent evals across all parallel experiments
_MATH_EVAL_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_MATH_EVALS)
_CODE_EVAL_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_CODE_EVALS)


def _get_no_think_datasets(task: str) -> set[str]:
    """Determine which datasets should NOT use think_prefix.

    Math-distilled models: no think_prefix on math evals (native thinking).
    Code-distilled models: no think_prefix on code evals (native thinking).
    """
    if task == "math":
        return MATH_DATASETS
    elif task == "code":
        return CODE_DATASETS
    return set()


async def _run_dataset(
    ds_name: str,
    sampling_client,
    renderer,
    tokenizer,
    results_dir: Path,
    model_label: str,
    think_prefix: bool,
    max_tokens: int,
    max_problems: int | None = None,
    **kwargs,
) -> tuple[str, dict]:
    """Run a single dataset evaluation."""
    print(f"\n  --- {ds_name} (think_prefix={think_prefix}, max_tokens={max_tokens}) ---")
    t0 = time.time()

    module = importlib.import_module(DATASET_MODULES[ds_name])
    extra_kwargs = dict(kwargs)
    if max_problems is not None:
        extra_kwargs["max_problems"] = max_problems

    result = await module.run(
        sampling_client,
        renderer,
        tokenizer,
        results_dir,
        model_label,
        think_prefix=think_prefix,
        max_tokens=max_tokens,
        **extra_kwargs,
    )
    elapsed = time.time() - t0
    print(f"  {ds_name} completed in {elapsed:.0f}s")
    return ds_name, result


async def _run_with_semaphore(
    ds_name: str,
    sampling_client,
    renderer,
    tokenizer,
    results_dir: Path,
    model_label: str,
    think_prefix: bool,
    max_tokens: int,
    serialize_code_eval: bool,
    max_problems: int | None = None,
    **kwargs,
) -> tuple[str, dict]:
    """Run an eval dataset with concurrency limiting via semaphores.

    Code datasets: limited by _CODE_EVAL_SEM (subprocess tests can OOM).
    Math/fuzzy datasets: limited by _MATH_EVAL_SEM (API throughput).
    """
    if ds_name in CODE_DATASETS and serialize_code_eval:
        sem = _CODE_EVAL_SEM
        label = "code eval"
    else:
        sem = _MATH_EVAL_SEM
        label = "math eval"

    loop = asyncio.get_event_loop()
    print(f"  Acquiring {label} semaphore for {ds_name}...")
    await loop.run_in_executor(None, sem.acquire)
    try:
        return await _run_dataset(
            ds_name, sampling_client, renderer, tokenizer,
            results_dir, model_label, think_prefix, max_tokens,
            max_problems=max_problems, **kwargs,
        )
    finally:
        sem.release()


async def evaluate_checkpoint(
    sampler_path: str,
    base_model: str,
    datasets: list[str],
    results_dir: Path,
    model_label: str,
    task: str,
    max_problems: int | None = None,
    max_tokens: int = 28672,
    serialize_code_eval: bool = True,
    **kwargs,
) -> dict[str, dict]:
    """Evaluate a fine-tuned checkpoint on specified datasets.

    Args:
        sampler_path: Tinker checkpoint URI.
        base_model: HF model ID (for renderer/tokenizer).
        datasets: List of dataset names to evaluate.
        results_dir: Where to save parquet results.
        model_label: Label for result files.
        task: "math" or "code" — determines think_prefix logic.
        max_problems: Limit problems per dataset (None = full).
        max_tokens: Max generation tokens.
        serialize_code_eval: Lock code evals to prevent OOM.

    Returns:
        Dict mapping dataset_name -> result_dict.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    renderer, tokenizer = setup_renderer_and_tokenizer(base_model)
    client = create_checkpoint_client(sampler_path)
    no_think = _get_no_think_datasets(task)

    valid_datasets = [ds for ds in datasets if ds in DATASET_MODULES]
    invalid = [ds for ds in datasets if ds not in DATASET_MODULES]
    for ds in invalid:
        print(f"  WARNING: Unknown dataset '{ds}'. Skipping.")

    # Run math/fuzzy datasets concurrently, code datasets with lock
    tasks = []
    for ds in valid_datasets:
        think_prefix = ds not in no_think
        tasks.append(_run_with_semaphore(
            ds, client, renderer, tokenizer,
            results_dir, model_label, think_prefix, max_tokens,
            serialize_code_eval, max_problems=max_problems, **kwargs,
        ))

    results_list = await asyncio.gather(*tasks)
    results = dict(results_list)

    # Save summary
    with open(results_dir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


async def evaluate_base_model(
    base_model: str,
    datasets: list[str],
    results_dir: Path,
    model_label: str,
    max_problems: int | None = None,
    max_tokens: int = 8192,
    serialize_code_eval: bool = True,
    **kwargs,
) -> dict[str, dict]:
    """Evaluate a base model (no LoRA). think_prefix=False on all datasets."""
    results_dir.mkdir(parents=True, exist_ok=True)
    renderer, tokenizer = setup_renderer_and_tokenizer(base_model)
    client = create_base_client(base_model)

    valid_datasets = [ds for ds in datasets if ds in DATASET_MODULES]

    tasks = []
    for ds in valid_datasets:
        tasks.append(_run_with_semaphore(
            ds, client, renderer, tokenizer,
            results_dir, model_label,
            think_prefix=False,
            max_tokens=max_tokens,
            serialize_code_eval=serialize_code_eval,
            max_problems=max_problems, **kwargs,
        ))

    results_list = await asyncio.gather(*tasks)
    results = dict(results_list)

    with open(results_dir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def select_best_checkpoint(
    checkpoint_results: dict[int, dict[str, dict]],
    task: str,
) -> tuple[int, str]:
    """Select the best checkpoint from mid-training eval results.

    Args:
        checkpoint_results: {step: {dataset_name: {accuracy: float, ...}}}
        task: "math" or "code" — determines primary metric.

    Returns:
        (best_step, reason_string)
    """
    if task == "math":
        primary = "math_500"
    else:
        primary = "mbppplus"

    candidates = []
    for step, results in checkpoint_results.items():
        primary_result = results.get(primary, {})
        acc = primary_result.get("accuracy")
        if acc is not None:
            candidates.append({"step": step, "accuracy": acc})

    if not candidates:
        # Fall back to any available metric
        for step, results in checkpoint_results.items():
            for ds, result in results.items():
                acc = result.get("accuracy")
                if acc is not None:
                    candidates.append({"step": step, "accuracy": acc})
                    break

    if not candidates:
        raise ValueError("No checkpoint results found with accuracy metric")

    best = max(candidates, key=lambda c: c["accuracy"])
    return best["step"], f"Best {primary}={best['accuracy']:.3f}"

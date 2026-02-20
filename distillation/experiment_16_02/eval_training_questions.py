"""Evaluate Qwen3-Base, Qwen3-Instruct, and Sonnet 4.5 on Sonnet training questions.

Samples 5000 questions from each training set (math/code), runs inference + grading.

Usage:
    # All models, both tasks
    PYTHONUNBUFFERED=1 python eval_training_questions.py

    # Single model, single task
    PYTHONUNBUFFERED=1 python eval_training_questions.py --models qwen_base --tasks math

    # Limit for testing
    PYTHONUNBUFFERED=1 python eval_training_questions.py --max-problems 20
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import pandas as pd

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, '/workspace/tinker-cookbook')

from config import (
    MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT,
    OPENROUTER_API_KEY, OPENROUTER_URL,
)
from infer import setup_renderer_and_tokenizer, create_base_client, generate
from utils import call_openrouter, extract_code
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

# ── Constants ────────────────────────────────────────────────────────────────

SONNET_MATH_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data" / "correct_only.jsonl"
SONNET_CODE_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data_code" / "correct_only.jsonl"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "training_question_evals"

SAMPLE_SIZE = 5000
SAMPLE_SEED = 42
MAX_TOKENS = 4096
CONCURRENCY = 10

QWEN_BASE = "Qwen/Qwen3-30B-A3B-Base"
QWEN_INSTRUCT = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SONNET_MODEL = "anthropic/claude-sonnet-4.5"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_math_sample(max_problems: int | None = None) -> list[dict]:
    """Load and sample math training questions."""
    with open(SONNET_MATH_DATA) as f:
        data = [json.loads(line) for line in f]

    rng = random.Random(SAMPLE_SEED)
    n = min(SAMPLE_SIZE, len(data))
    if max_problems:
        n = min(n, max_problems)
    sample = rng.sample(data, n)

    return [{"problem": d["problem"], "answer": d["answer"], "uuid": d.get("uuid", str(i))}
            for i, d in enumerate(sample)]


def load_code_sample(max_problems: int | None = None) -> list[dict]:
    """Load and sample code training questions."""
    with open(SONNET_CODE_DATA) as f:
        data = [json.loads(line) for line in f]

    rng = random.Random(SAMPLE_SEED)
    n = min(SAMPLE_SIZE, len(data))
    if max_problems:
        n = min(n, max_problems)
    sample = rng.sample(data, n)

    return [{"question": d["question"], "test": d["test"], "question_id": d.get("question_id", str(i))}
            for i, d in enumerate(sample)]


# ── Math grading ─────────────────────────────────────────────────────────────

def grade_math_response(raw_output: str, ground_truth: str) -> tuple[bool, str]:
    """Grade a math response. Returns (correct, predicted_answer)."""
    try:
        predicted = extract_boxed(raw_output)
    except (ValueError, Exception):
        predicted = None
    if predicted is None or predicted == "no_boxed":
        return False, "no_boxed"

    try:
        if grade_answer(predicted, ground_truth):
            return True, predicted
    except Exception:
        pass

    try:
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify
        if grade_answer_math_verify(predicted, ground_truth):
            return True, predicted
    except Exception:
        pass

    return False, predicted


# ── Code grading ─────────────────────────────────────────────────────────────

def run_code_test(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run pytest-style tests on extracted code. Returns (passed, detail)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"
        sol_path.write_text(code)
        test_path.write_text(test_code)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
            passed = result.returncode == 0
            detail = "passed" if passed else result.stdout[-500:] + result.stderr[-500:]
            return passed, detail
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, f"error: {e}"


def grade_code_response(raw_output: str, test_code: str) -> tuple[bool, str]:
    """Grade a code response by extracting code and running tests."""
    code = extract_code(raw_output)
    if not code:
        return False, "no_code_extracted"
    return run_code_test(code, test_code)


# ── Qwen inference ───────────────────────────────────────────────────────────

async def run_qwen_eval(
    model_name: str,
    problems: list[dict],
    task: str,  # "math" or "code"
    results_name: str,
):
    """Run eval on Qwen model (base or instruct) via tinker."""
    print(f"\n{'='*60}")
    print(f"  {results_name}: {len(problems)} {task} problems")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    renderer, tokenizer = setup_renderer_and_tokenizer(model_name)
    client = create_base_client(model_name)

    system_prompt = MATH_SYSTEM_PROMPT if task == "math" else CODE_SYSTEM_PROMPT
    # Instruct model doesn't need think_prefix; base model does
    think_prefix = "Instruct" not in model_name

    results = []
    t0 = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i: int, prob: dict):
        question = prob["problem"] if task == "math" else prob["question"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        async with sem:
            completions = await generate(
                client, renderer, tokenizer,
                messages=messages,
                max_tokens=MAX_TOKENS,
                think_prefix=think_prefix,
            )

        raw_output = completions[0] if completions else ""

        if task == "math":
            correct, predicted = grade_math_response(raw_output, prob["answer"])
            return {
                "problem_id": prob.get("uuid", str(i)),
                "question": question,
                "raw_output": raw_output,
                "correct": correct,
                "predicted_answer": predicted,
                "ground_truth": prob["answer"],
            }
        else:
            passed, detail = grade_code_response(raw_output, prob["test"])
            return {
                "problem_id": prob.get("question_id", str(i)),
                "question": question,
                "raw_output": raw_output,
                "passed": passed,
                "detail": detail,
            }

    tasks = [process_one(i, p) for i, p in enumerate(problems)]

    # Process in chunks for progress reporting
    chunk_size = max(1, len(tasks) // 10)
    for start in range(0, len(tasks), chunk_size):
        chunk = tasks[start:start + chunk_size]
        chunk_results = await asyncio.gather(*chunk)
        results.extend(chunk_results)
        elapsed = time.time() - t0
        n_done = len(results)
        correct_col = "correct" if task == "math" else "passed"
        n_correct = sum(1 for r in results if r.get(correct_col, False))
        print(f"  [{results_name}] {n_done}/{len(problems)} done, "
              f"{n_correct} correct ({n_correct/n_done*100:.1f}%) [{elapsed:.0f}s]")

    # Save
    df = pd.DataFrame(results)
    df["model"] = results_name
    df["task"] = task
    out_dir = RESULTS_DIR / results_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_training_{task}.parquet"
    df.to_parquet(out_path, index=False)

    correct_col = "correct" if task == "math" else "passed"
    acc = df[correct_col].mean() * 100
    print(f"  [{results_name}] FINAL: {acc:.1f}% ({df[correct_col].sum()}/{len(df)})")
    print(f"  Saved to {out_path}")
    return df


# ── Sonnet 4.5 inference via OpenRouter ──────────────────────────────────────

async def run_sonnet_eval(
    problems: list[dict],
    task: str,
    results_name: str = "sonnet45",
):
    """Run eval on Sonnet 4.5 via OpenRouter."""
    print(f"\n{'='*60}")
    print(f"  {results_name}: {len(problems)} {task} problems")
    print(f"  Model: {SONNET_MODEL}")
    print(f"{'='*60}")

    system_prompt = MATH_SYSTEM_PROMPT if task == "math" else CODE_SYSTEM_PROMPT
    sem = asyncio.Semaphore(CONCURRENCY)

    results = []
    completed = 0
    t0 = time.time()
    report_interval = max(1, len(problems) // 10)

    async with aiohttp.ClientSession() as session:
        async def process_one(i: int, prob: dict):
            nonlocal completed
            question = prob["problem"] if task == "math" else prob["question"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            raw_output = await call_openrouter(
                session=session,
                model=SONNET_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                semaphore=sem,
                api_key=OPENROUTER_API_KEY,
            )

            if raw_output is None or (isinstance(raw_output, str) and raw_output.startswith("ERROR:")):
                raw_output = raw_output or ""

            if task == "math":
                correct, predicted = grade_math_response(raw_output, prob["answer"])
                row = {
                    "problem_id": prob.get("uuid", str(i)),
                    "question": question,
                    "raw_output": raw_output,
                    "correct": correct,
                    "predicted_answer": predicted,
                    "ground_truth": prob["answer"],
                }
            else:
                passed, detail = grade_code_response(raw_output, prob["test"])
                row = {
                    "problem_id": prob.get("question_id", str(i)),
                    "question": question,
                    "raw_output": raw_output,
                    "passed": passed,
                    "detail": detail,
                }

            completed += 1
            if completed % report_interval == 0 or completed == len(problems):
                elapsed = time.time() - t0
                correct_col = "correct" if task == "math" else "passed"
                # Can't easily count all correct here in async, just report count
                print(f"  [{results_name}] {completed}/{len(problems)} done [{elapsed:.0f}s]")

            return row

        tasks = [process_one(i, p) for i, p in enumerate(problems)]
        results = await asyncio.gather(*tasks)

    # Save
    df = pd.DataFrame(list(results))
    df["model"] = results_name
    df["task"] = task
    out_dir = RESULTS_DIR / results_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_training_{task}.parquet"
    df.to_parquet(out_path, index=False)

    correct_col = "correct" if task == "math" else "passed"
    acc = df[correct_col].mean() * 100
    print(f"  [{results_name}] FINAL: {acc:.1f}% ({df[correct_col].sum()}/{len(df)})")
    print(f"  Saved to {out_path}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen_base,qwen_instruct,sonnet45",
                        help="Comma-separated: qwen_base,qwen_instruct,sonnet45")
    parser.add_argument("--tasks", default="math,code",
                        help="Comma-separated: math,code")
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Load samples
    math_sample, code_sample = None, None
    if "math" in tasks:
        print("Loading math training sample...")
        math_sample = load_math_sample(args.max_problems)
        print(f"  {len(math_sample)} math problems sampled")
    if "code" in tasks:
        print("Loading code training sample...")
        code_sample = load_code_sample(args.max_problems)
        print(f"  {len(code_sample)} code problems sampled")

    # Run evals sequentially per model (each model uses full concurrency)
    for model_key in models:
        for task in tasks:
            sample = math_sample if task == "math" else code_sample
            if sample is None:
                continue

            if model_key == "qwen_base":
                await run_qwen_eval(QWEN_BASE, sample, task, "qwen_base")
            elif model_key == "qwen_instruct":
                await run_qwen_eval(QWEN_INSTRUCT, sample, task, "qwen_instruct")
            elif model_key == "sonnet45":
                await run_sonnet_eval(sample, task, "sonnet45")
            else:
                print(f"Unknown model: {model_key}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

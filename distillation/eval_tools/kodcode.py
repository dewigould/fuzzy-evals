"""KodCode evaluation: 500 easy Python problems."""

import asyncio
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_TOKENS, EVAL_CONCURRENCY, CODE_SYSTEM_PROMPT
from infer import generate
from utils import parse_think_tags, extract_code_from_response, save_results_parquet

N_PROBLEMS = 500
HARD_CAP_SECONDS = 30
PER_TEST_TIMEOUT = 4
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


@dataclass
class KodCodeTask:
    question: str
    test_code: str
    gpt_difficulty: str
    gpt_pass_pct: float


def load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42) -> list[KodCodeTask]:
    """Load KodCode tasks, filtered by difficulty. Python only, no numpy/pandas/torch."""
    import random
    print(f"Loading KodCode tasks (difficulty={difficulty}, max={max_tasks})")
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    tasks = []
    for row in ds:
        if row["gpt_difficulty"] != difficulty:
            continue
        test = row["test"]
        if "from solution import" not in test:
            continue
        test_lower = test.lower()
        if any(lib in test_lower for lib in ["import numpy", "import pandas", "import torch"]):
            continue
        tasks.append(KodCodeTask(
            question=row["question"],
            test_code=test,
            gpt_difficulty=row["gpt_difficulty"],
            gpt_pass_pct=row["gpt_pass_percentage"],
        ))
        if len(tasks) >= max_tasks:
            break

    print(f"  Loaded {len(tasks)} KodCode tasks")
    random.Random(seed).shuffle(tasks)
    return tasks


def run_kodcode_tests_sync(code: str, test_code: str, timeout: int = 10) -> bool:
    """Run code against KodCode pytest-style tests in isolated subprocess."""
    resource_wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
    )

    runner = resource_wrapper + '''
import sys
import importlib.util
import traceback

sys.path.insert(0, ".")

try:
    import solution

    spec = importlib.util.spec_from_file_location("test_solution", "test_solution.py")
    test_mod = importlib.util.module_from_spec(spec)

    for name in dir(solution):
        if not name.startswith('_'):
            setattr(test_mod, name, getattr(solution, name))

    spec.loader.exec_module(test_mod)

    test_funcs = [name for name in dir(test_mod) if name.startswith('test_')]
    failed = 0
    for name in test_funcs:
        try:
            getattr(test_mod, name)()
        except Exception:
            failed += 1

    if failed > 0 or len(test_funcs) == 0:
        sys.exit(1)
    else:
        sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(test_code)
        with open(os.path.join(tmpdir, "run_tests.py"), "w") as f:
            f.write(runner)

        try:
            result = subprocess.run(
                [sys.executable, "run_tests.py"],
                cwd=tmpdir, capture_output=True,
                timeout=timeout + 2, text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on KodCode-500."""
    all_tasks = load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42)
    # Take last 500 as test set (first N used for training in RLVR)
    test_problems = all_tasks[-N_PROBLEMS:]
    if max_problems is not None:
        test_problems = test_problems[:max_problems]
    print(f"[kodcode] Evaluating {model_name} on {len(test_problems)} problems")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    loop = asyncio.get_event_loop()
    results = []
    counter = {"done": 0, "correct": 0}
    t0 = time.time()

    async def eval_one(idx, problem):
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": problem.question},
        ]
        async with semaphore:
            try:
                resp = await generate(
                    sampling_client, renderer, tokenizer,
                    messages=messages, max_tokens=max_tokens,
                    temperature=0.0, think_prefix=think_prefix,
                )
                text = resp[0]
            except Exception as e:
                text = f"ERROR: {e}"

        cot, user_output = parse_think_tags(text)
        code = extract_code_from_response(text)
        passed = False
        detail = "no code extracted"
        if code is not None:
            passed = await loop.run_in_executor(
                _test_executor, run_kodcode_tests_sync, code, problem.test_code,
            )
            detail = "passed" if passed else "failed"

        counter["done"] += 1
        if passed:
            counter["correct"] += 1
        if counter["done"] % 50 == 0 or counter["done"] == len(test_problems):
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [kodcode] {counter['done']}/{len(test_problems)} done, "
                  f"{counter['correct']} correct ({pct:.1f}%) [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "kodcode_500",
            "question": problem.question[:500],
            "raw_output": text,
            "cot": cot,
            "user_output": user_output,
            "passed": passed,
            "detail": detail,
        }

    # Process in batches to avoid overwhelming the test executor
    for batch_start in range(0, len(test_problems), 20):
        batch_end = min(batch_start + 20, len(test_problems))
        batch_results = await asyncio.gather(*[
            eval_one(i, test_problems[i])
            for i in range(batch_start, batch_end)
        ])
        results.extend(batch_results)

    n_correct = sum(1 for r in results if r["passed"])
    accuracy = n_correct / len(test_problems)
    print(f"[kodcode] FINAL: {n_correct}/{len(test_problems)} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_kodcode_500.parquet")
    return {"dataset": "kodcode_500", "accuracy": accuracy, "n_correct": n_correct, "total": len(test_problems)}

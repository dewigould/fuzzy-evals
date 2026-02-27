"""HumanEval+ evaluation: 164 Python function completion problems.

Uses evalplus/humanevalplus from HuggingFace (80x more tests than original HumanEval).
Model receives function signature + docstring, must complete the function.
Tests use check(candidate) pattern with numpy-based assertions.
"""

import asyncio
import concurrent.futures
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_TOKENS, EVAL_CONCURRENCY, CODE_SYSTEM_PROMPT
from infer import generate
from utils import parse_think_tags, extract_code_from_response, save_results_parquet, load_or_build_cache

EVAL_SEED = 42
HARD_CAP_SECONDS = 30
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def _build_humanevalplus_cache() -> list[dict]:
    """Load all HumanEval+ problems, shuffled with seed."""
    print("  Loading HumanEval+ from HuggingFace...")
    ds = load_dataset("evalplus/humanevalplus", split="test")
    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
        })
    random.Random(EVAL_SEED).shuffle(problems)
    print(f"  Loaded {len(problems)} HumanEval+ problems")
    return problems


def run_humanevalplus_test_sync(code: str, test_code: str, entry_point: str,
                                 timeout: int = HARD_CAP_SECONDS) -> tuple[bool, str]:
    """Run code against HumanEval+ tests in isolated subprocess.

    The test_code defines check(candidate) which calls the function with various inputs.
    We write: model_code + test_code + check(entry_point)
    """
    # Build the full test file
    full_code = (
        f"{code}\n\n"
        f"{test_code}\n\n"
        f"check({entry_point})\n"
    )

    # Resource limits wrapper
    resource_wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        # RLIMIT_NPROC omitted: numpy/OpenBLAS needs threads
    )

    runner = resource_wrapper + """
import sys
import traceback

try:
    exec(open("solution.py").read())
    sys.exit(0)
except AssertionError as e:
    print(f"ASSERTION: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(full_code)
        with open(os.path.join(tmpdir, "run_tests.py"), "w") as f:
            f.write(runner)

        try:
            result = subprocess.run(
                [sys.executable, "run_tests.py"],
                cwd=tmpdir, capture_output=True,
                timeout=timeout + 5, text=True,
            )
            if result.returncode == 0:
                return True, "passed"
            else:
                detail = result.stderr.strip()[-300:] if result.stderr else f"exit code {result.returncode}"
                return False, detail
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on HumanEval+ (164 problems)."""
    cached = load_or_build_cache("humanevalplus", _build_humanevalplus_cache, EVAL_SEED)
    problems = cached
    if max_problems is not None:
        problems = problems[:max_problems]
    total = len(problems)
    print(f"[humanevalplus] Evaluating {model_name} on {total} problems")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    loop = asyncio.get_event_loop()
    results = []
    counter = {"done": 0, "correct": 0}
    t0 = time.time()

    async def eval_one(idx):
        problem = problems[idx]
        # The prompt contains the function signature + docstring
        # Ask model to complete it
        user_prompt = (
            f"Complete the following Python function:\n\n"
            f"```python\n{problem['prompt']}```"
        )
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
            passed, detail = await loop.run_in_executor(
                _test_executor, run_humanevalplus_test_sync,
                code, problem["test"], problem["entry_point"],
            )

        counter["done"] += 1
        if passed:
            counter["correct"] += 1
        if counter["done"] % 50 == 0 or counter["done"] == total:
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [humanevalplus] {counter['done']}/{total} done, "
                  f"{counter['correct']} correct ({pct:.1f}%) [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "humanevalplus",
            "task_id": problem["task_id"],
            "entry_point": problem["entry_point"],
            "raw_output": text,
            "cot": cot,
            "user_output": user_output,
            "passed": passed,
            "detail": detail,
        }

    # Process in batches
    for batch_start in range(0, total, 20):
        batch_end = min(batch_start + 20, total)
        batch_results = await asyncio.gather(*[
            eval_one(i) for i in range(batch_start, batch_end)
        ])
        results.extend(batch_results)

    n_correct = sum(1 for r in results if r["passed"])
    accuracy = n_correct / total
    print(f"[humanevalplus] FINAL: {n_correct}/{total} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_humanevalplus.parquet")
    return {"dataset": "humanevalplus", "accuracy": accuracy, "n_correct": n_correct, "total": total}

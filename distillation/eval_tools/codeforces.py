"""Codeforces evaluation: 500 problems from DeepCoder dataset."""

import asyncio
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/workspace/tinker-cookbook')

from config import MAX_TOKENS, EVAL_CONCURRENCY, CODE_SYSTEM_PROMPT
from infer import generate
from utils import parse_think_tags, extract_code_from_response, save_results_parquet

from tinker_cookbook.recipes.code_rl.lcb_utils import (
    fetch_live_code_bench_system_prompt,
    TEST_UTIL as _TEST_UTIL_ORIG,
    TEST_CODE,
)

# Patch TEST_UTIL for sys.stdin.buffer.readline() compat
TEST_UTIL = _TEST_UTIL_ORIG.replace(
    "from io import StringIO",
    "from io import StringIO, BytesIO\n\n"
    "class BufferedStringIO(StringIO):\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        super().__init__(*args, **kwargs)\n"
    "        self.buffer = BytesIO(self.getvalue().encode())\n",
).replace(
    "StringIO(inputs)",
    "BufferedStringIO(inputs)",
)

N_PROBLEMS = 500
TIMEOUT_PER_TEST = 6
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _normalize_tests(raw_tests, metadata) -> list[dict]:
    """Normalize test cases to unified format."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    tests = raw_tests
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        inputs = tests.get("inputs", [])
        outputs = tests.get("outputs", [])
        converted = []
        for inp, out in zip(inputs, outputs):
            if isinstance(out, list):
                out = out[0] if out else ""
            converted.append({
                "input": inp, "output": out,
                "testtype": "functional" if "fn_name" in tests else "stdin_stdout",
                "metadata": {"func_name": tests["fn_name"]} if "fn_name" in tests else {},
            })
        tests = converted
    if isinstance(tests, dict):
        tests = [tests]

    normalized = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = test.get("metadata", {})
        if isinstance(test_metadata, str):
            try:
                test_metadata = json.loads(test_metadata)
            except json.JSONDecodeError:
                test_metadata = {}
        if not isinstance(test_metadata, dict):
            test_metadata = {}
        normalized.append({
            "input": str(test.get("input", "")),
            "output": str(test.get("output", "")),
            "testtype": testtype,
            "metadata": test_metadata or {"func_name": None},
        })
    return normalized


def postprocess_lcb_sample(sample: list[dict]) -> dict:
    """Convert test cases to format expected by run_test()."""
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]
    sample_dict = {"inputs": sample_inputs, "outputs": sample_outputs}
    if sample[0].get("testtype") == "functional":
        fn_name = sample[0].get("metadata", {}).get("func_name")
        if fn_name:
            sample_dict["fn_name"] = fn_name
    return {"input_output": json.dumps(sample_dict)}


def load_codeforces_problems(n: int) -> list[dict]:
    """Load first n problems from the codeforces test split (streaming)."""
    print(f"Loading first {n} Codeforces problems (streaming)...")
    ds = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        name="codeforces",
        split="test",
        streaming=True,
    )
    problems = []
    for i, row in enumerate(ds):
        if i >= n * 2:  # scan extra to fill quota
            break
        question = row.get("question") or row.get("prompt") or row.get("problem")
        if not question:
            continue

        raw_tests = row.get("tests") or row.get("ground_truth")
        if isinstance(raw_tests, str):
            raw_tests = json.loads(raw_tests)
        tests = _normalize_tests(raw_tests, row.get("metadata", {}))
        if not tests:
            continue

        starter_code = row.get("starter_code")
        if isinstance(starter_code, str) and starter_code.strip():
            prompt = fetch_live_code_bench_system_prompt(question, starter_code)
        else:
            prompt = fetch_live_code_bench_system_prompt(question)
            starter_code = None

        problems.append({
            "prompt": prompt,
            "question_text": question[:500],
            "tests": tests,
            "starter_code": starter_code,
        })
        if len(problems) >= n:
            break

    print(f"Loaded {len(problems)} valid problems")
    return problems


def run_test_subprocess(test_cases: dict, code: str, timeout: int = 6) -> tuple[bool, str]:
    """Run code against test cases in an isolated subprocess."""
    test_code_str = TEST_CODE % {"timeout": timeout}
    n_tests = len(json.loads(test_cases["input_output"])["inputs"])
    total_timeout = (timeout + 1) * n_tests + 5

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test_cases.txt"), "w") as f:
            json.dump(test_cases, f)
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "testing_util.py"), "w") as f:
            f.write(TEST_UTIL)
        with open(os.path.join(tmpdir, "run.py"), "w") as f:
            f.write(test_code_str)

        try:
            result = subprocess.run(
                [sys.executable, "run.py"],
                capture_output=True, text=True,
                timeout=min(total_timeout, 60),
                cwd=tmpdir,
            )
            if result.returncode == 0:
                return True, "all tests passed"
            else:
                detail = result.stdout.strip()[-200:] if result.stdout else result.stderr.strip()[-200:]
                return False, detail or f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]


async def run(sampling_client, renderer, tokenizer, results_dir: Path, model_name: str,
              think_prefix: bool = True, max_tokens: int = MAX_TOKENS,
              max_problems: int | None = None, **kwargs) -> dict:
    """Evaluate on Codeforces-500."""
    n = max_problems if max_problems is not None else N_PROBLEMS
    problems = load_codeforces_problems(n)
    print(f"[codeforces] Evaluating {model_name} on {len(problems)} problems")

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    loop = asyncio.get_event_loop()
    results = []
    counter = {"done": 0, "correct": 0}
    t0 = time.time()

    async def eval_one(i):
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": problems[i]["prompt"]},
        ]
        async with semaphore:
            try:
                completions = await generate(
                    sampling_client, renderer, tokenizer,
                    messages=messages, max_tokens=max_tokens,
                    temperature=0.0, think_prefix=think_prefix,
                )
                completion = completions[0]
            except Exception as e:
                completion = f"ERROR: {e}"

        cot, user_output = parse_think_tags(completion)
        code = extract_code_from_response(completion)
        passed = False
        detail = "no code extracted"

        if code is not None:
            test_cases = postprocess_lcb_sample(problems[i]["tests"])
            passed, detail = await loop.run_in_executor(
                _test_executor, run_test_subprocess, test_cases, code, TIMEOUT_PER_TEST,
            )

        counter["done"] += 1
        if passed:
            counter["correct"] += 1
        if counter["done"] % 50 == 0 or counter["done"] == len(problems):
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [codeforces] {counter['done']}/{len(problems)} done, "
                  f"{counter['correct']} correct ({pct:.1f}%) [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "codeforces_500",
            "question": problems[i]["question_text"],
            "raw_output": completion,
            "cot": cot,
            "user_output": user_output,
            "passed": passed,
            "detail": detail,
        }

    # Process in batches
    for batch_start in range(0, len(problems), 20):
        batch_end = min(batch_start + 20, len(problems))
        batch_results = await asyncio.gather(*[
            eval_one(i) for i in range(batch_start, batch_end)
        ])
        results.extend(batch_results)

    n_passed = sum(1 for r in results if r["passed"])
    accuracy = n_passed / len(problems)
    print(f"[codeforces] FINAL: {n_passed}/{len(problems)} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_codeforces_500.parquet")
    return {"dataset": "codeforces_500", "accuracy": accuracy, "n_correct": n_passed, "total": len(problems)}

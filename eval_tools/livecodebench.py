"""LiveCodeBench evaluation: code_generation_lite release_v5 (167 problems).

Uses livecodebench/code_generation_lite from HuggingFace.
Two test types: stdin (atcoder) and functional (leetcode with starter code).
Test execution reuses the same infrastructure as the codeforces module.
"""

import asyncio
import base64
import concurrent.futures
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

EVAL_SEED = 42

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/workspace/tinker-cookbook')

from config import MAX_TOKENS, EVAL_CONCURRENCY, CODE_SYSTEM_PROMPT
from infer import generate
from utils import parse_think_tags, extract_code_from_response, save_results_parquet, load_or_build_cache

from tinker_cookbook.recipes.code_rl.lcb_utils import (
    fetch_live_code_bench_system_prompt,
    LCB_SYSTEM_MESSAGE_GENERIC,
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

TIMEOUT_PER_TEST = 6
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _decode_private_tests(encoded: str) -> list[dict]:
    """Decode base64 -> zlib -> pickle -> JSON private test cases."""
    raw = base64.b64decode(encoded)
    decompressed = zlib.decompress(raw)
    json_str = pickle.loads(decompressed)
    return json.loads(json_str)


def _build_livecodebench_order() -> list[dict]:
    """Build a lightweight cache: just the shuffled order (indices) + metadata."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        'livecodebench/code_generation_lite', 'test5.jsonl', repo_type='dataset'
    )
    with open(path) as f:
        n_rows = sum(1 for _ in f)

    indices = list(range(n_rows))
    random.Random(EVAL_SEED).shuffle(indices)
    # Store just the shuffled order — actual data loaded from HF-cached JSONL
    return [{"original_index": i} for i in indices]


def load_livecodebench_problems(max_problems: int | None = None) -> list[dict]:
    """Load LiveCodeBench v5 problems in shuffled order from HF-cached JSONL."""
    from huggingface_hub import hf_hub_download

    # Get shuffled order from cache
    order = load_or_build_cache("livecodebench_v5", _build_livecodebench_order, EVAL_SEED)
    if max_problems is not None:
        order = order[:min(max_problems, len(order))]

    # Load raw data from HF-cached JSONL
    path = hf_hub_download(
        'livecodebench/code_generation_lite', 'test5.jsonl', repo_type='dataset'
    )
    with open(path) as f:
        all_rows = [json.loads(line) for line in f]

    # Build problems in shuffled order, decoding tests
    problems = []
    for item in order:
        row = all_rows[item["original_index"]]
        public_tests = json.loads(row['public_test_cases'])
        private_tests = _decode_private_tests(row['private_test_cases'])

        starter_code = row.get('starter_code', '').strip() or None
        if starter_code:
            prompt = fetch_live_code_bench_system_prompt(
                row['question_content'], starter_code
            )
        else:
            prompt = fetch_live_code_bench_system_prompt(row['question_content'])
        # Strip the "You are an expert Python programmer..." preamble —
        # we use our own CODE_SYSTEM_PROMPT as the system message instead
        prompt = prompt.removeprefix(LCB_SYSTEM_MESSAGE_GENERIC).lstrip("\n")

        problems.append({
            'prompt': prompt,
            'title': row['question_title'],
            'difficulty': row['difficulty'],
            'platform': row['platform'],
            'tests': public_tests + private_tests,
            'starter_code': starter_code,
        })

    print(f"Selected {len(problems)} LiveCodeBench v5 problems")
    return problems


def _postprocess_tests(tests: list[dict]) -> dict:
    """Convert test cases to the format expected by the test runner."""
    inputs = [t['input'] for t in tests]
    outputs = [t['output'] for t in tests]
    sample_dict = {"inputs": inputs, "outputs": outputs}

    if tests[0].get('testtype') == 'functional':
        # Extract function name from starter code or test structure
        # For functional tests, the runner needs fn_name
        sample_dict["fn_name"] = None  # testing_util handles this for class methods
    return {"input_output": json.dumps(sample_dict)}


def _run_test_subprocess(test_cases: dict, code: str, timeout: int = 6) -> tuple[bool, str]:
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
    """Evaluate on LiveCodeBench v5."""
    problems = load_livecodebench_problems(max_problems)
    total = len(problems)
    print(f"[livecodebench] Evaluating {model_name} on {total} problems")

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
            test_cases = _postprocess_tests(problems[i]["tests"])
            passed, detail = await loop.run_in_executor(
                _test_executor, _run_test_subprocess, test_cases, code, TIMEOUT_PER_TEST,
            )

        counter["done"] += 1
        if passed:
            counter["correct"] += 1
        if counter["done"] % 50 == 0 or counter["done"] == total:
            elapsed = time.time() - t0
            pct = counter["correct"] / counter["done"] * 100
            print(f"  [livecodebench] {counter['done']}/{total} done, "
                  f"{counter['correct']} correct ({pct:.1f}%) [{elapsed:.0f}s]")

        return {
            "model": model_name,
            "dataset": "livecodebench_v5",
            "question": problems[i]["title"],
            "difficulty": problems[i]["difficulty"],
            "platform": problems[i]["platform"],
            "raw_output": completion,
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

    n_passed = sum(1 for r in results if r["passed"])
    accuracy = n_passed / total
    print(f"[livecodebench] FINAL: {n_passed}/{total} ({accuracy*100:.1f}%)")

    save_results_parquet(results, results_dir / "results_livecodebench_v5.parquet")
    return {"dataset": "livecodebench_v5", "accuracy": accuracy, "n_correct": n_passed, "total": total}

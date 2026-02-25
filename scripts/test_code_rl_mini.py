"""
Minimal timing test: generate code for 3 problems, test with original vs capped timeout.
"""

import asyncio
import json
import sys
import time
import concurrent.futures

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample

from datasets import load_dataset
from tinker_cookbook.recipes.code_rl.code_env import (
    _build_question, _ensure_dict, _normalize_tests, DeepcoderTask,
)
from run_code_rlvr import extract_code, run_code_tests_sync, CodeProblemEnv
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL

import subprocess, tempfile, os

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOTAL_TIMEOUT = 30  # Cap


def run_code_tests_capped(code, tests, timeout=6, max_total=MAX_TOTAL_TIMEOUT):
    """Like run_code_tests_sync but caps total_timeout."""
    test_cases = postprocess_lcb_sample(tests)
    test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
    original_total = (timeout + 1) * test_cnt + 5
    total_timeout = min(original_total, max_total)

    resource_wrapper = (
        "import resource, signal\n"
        "resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({total_timeout}, {total_timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
        f"signal.alarm({total_timeout})\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test_cases.txt"), "w") as f:
            f.write(json.dumps(test_cases))
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "testing_util.py"), "w") as f:
            f.write(TEST_UTIL)
        with open(os.path.join(tmpdir, "run.py"), "w") as f:
            f.write(resource_wrapper + (TEST_CODE % {"timeout": timeout}))

        try:
            result = subprocess.run(
                [sys.executable, "run.py"],
                cwd=tmpdir,
                capture_output=True,
                timeout=total_timeout + 2,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def load_problems(n=3):
    print(f"Loading {n} DeepCoder problems...")
    ds = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        name="primeintellect", split="train", streaming=True,
    )
    tasks = []
    for row in ds:
        metadata = _ensure_dict(row.get("metadata", {}))
        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_tests(raw_tests, metadata)
        if not tests:
            continue
        problem = _build_question(row)
        if problem is None:
            continue
        tasks.append(DeepcoderTask(problem=problem, tests=tests, starter_code=None))
        if len(tasks) >= n:
            break
    del ds
    return tasks


async def main():
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(base_model=MODEL_NAME)

    tasks = load_problems(3)

    # Show test counts
    print("\n=== Test case analysis ===")
    for i, task in enumerate(tasks):
        tc = postprocess_lcb_sample(task.tests)
        n = len(json.loads(tc["input_output"])["inputs"])
        orig = (6 + 1) * n + 5
        capped = min(orig, MAX_TOTAL_TIMEOUT)
        print(f"  Problem {i}: {n} tests, orig_timeout={orig}s, capped={capped}s")

    # Generate code for each
    print("\n=== Generation ===")
    codes = []
    for i, task in enumerate(tasks):
        question = task.problem + "\n\nProvide your solution in a ```python code block."
        messages = [{"role": "user", "content": question}]
        model_input = renderer.build_generation_prompt(messages)
        t0 = time.time()
        resp = await client.sample_async(
            model_input, num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=2048, temperature=1.0,
                                                   stop=renderer.get_stop_sequences()),
        )
        elapsed = time.time() - t0
        parsed, _ = renderer.parse_response(resp.sequences[0].tokens)
        text = parsed['content']
        code = extract_code(text)
        codes.append(code)
        print(f"  Problem {i}: {len(resp.sequences[0].tokens)} tokens, {elapsed:.1f}s, has_code={code is not None}")

    # Test with CAPPED timeout
    print("\n=== Testing with capped timeout (30s max) ===")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_event_loop()
    for i, (task, code) in enumerate(zip(tasks, codes)):
        if code is None:
            print(f"  Problem {i}: SKIP (no code)")
            continue
        t0 = time.time()
        passed = await loop.run_in_executor(executor, run_code_tests_capped, code, task.tests)
        elapsed = time.time() - t0
        print(f"  Problem {i}: passed={passed}, time={elapsed:.1f}s")

    # Test with ORIGINAL timeout for comparison
    print("\n=== Testing with ORIGINAL timeout (for comparison) ===")
    for i, (task, code) in enumerate(zip(tasks, codes)):
        if code is None:
            print(f"  Problem {i}: SKIP (no code)")
            continue
        t0 = time.time()
        passed = await loop.run_in_executor(executor, run_code_tests_sync, code, task.tests, 6)
        elapsed = time.time() - t0
        print(f"  Problem {i}: passed={passed}, time={elapsed:.1f}s")

    executor.shutdown(wait=False)
    print("\nDONE")


if __name__ == '__main__':
    asyncio.run(main())

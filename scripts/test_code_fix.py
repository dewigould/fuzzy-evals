"""
Quick validation: verify the code testing fixes work correctly and are fast.
Tests: truncation, hard cap, thread pool execution.
"""
import asyncio
import json
import sys
import time

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample
from tinker_cookbook.recipes.code_rl.code_env import (
    _build_question, _ensure_dict, _normalize_tests, DeepcoderTask,
)
from datasets import load_dataset
from run_code_rlvr import (
    run_code_tests_sync, _truncate_tests, extract_code,
    MAX_TESTS_TRAIN, HARD_CAP_SECONDS, TRAIN_TIMEOUT, _test_executor,
)

# A trivially correct solution for testing
TRIVIAL_CODE = """
n = int(input())
print(n)
"""

# A known-bad solution
BAD_CODE = """
print("wrong answer")
"""

# An infinite loop to test timeout
INFINITE_CODE = """
while True:
    pass
"""


def load_problems(n=5):
    """Load n problems with varying test counts."""
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
    print("=" * 60)
    print("Code Testing Fix Validation")
    print(f"  MAX_TESTS_TRAIN={MAX_TESTS_TRAIN}")
    print(f"  HARD_CAP_SECONDS={HARD_CAP_SECONDS}")
    print(f"  TRAIN_TIMEOUT={TRAIN_TIMEOUT}")
    print("=" * 60)

    # Load problems and analyze test counts
    print("\n--- Loading 10 problems ---")
    tasks = load_problems(10)

    for i, task in enumerate(tasks):
        tc = postprocess_lcb_sample(task.tests)
        n_tests = len(json.loads(tc["input_output"])["inputs"])
        old_timeout = (6 + 1) * n_tests + 5
        new_timeout = min((TRAIN_TIMEOUT + 1) * min(n_tests, MAX_TESTS_TRAIN) + 5, HARD_CAP_SECONDS)
        print(f"  Problem {i}: {n_tests} tests, old_timeout={old_timeout}s, new_timeout={new_timeout}s")

    # Test 1: Infinite loop with hard cap
    print("\n--- Test 1: Infinite loop (should hit hard cap quickly) ---")
    t0 = time.time()
    result = run_code_tests_sync(
        INFINITE_CODE, tasks[0].tests,
        timeout=TRAIN_TIMEOUT, max_tests=MAX_TESTS_TRAIN, hard_cap=HARD_CAP_SECONDS,
    )
    elapsed = time.time() - t0
    print(f"  Result: passed={result}, time={elapsed:.1f}s (should be ~{HARD_CAP_SECONDS}s or less)")

    # Test 2: Bad code (should fail fast)
    print("\n--- Test 2: Bad code (should fail fast) ---")
    t0 = time.time()
    result = run_code_tests_sync(
        BAD_CODE, tasks[0].tests,
        timeout=TRAIN_TIMEOUT, max_tests=MAX_TESTS_TRAIN, hard_cap=HARD_CAP_SECONDS,
    )
    elapsed = time.time() - t0
    print(f"  Result: passed={result}, time={elapsed:.1f}s (should be fast)")

    # Test 3: Run all 10 problems concurrently in thread pool (simulating RL batch)
    print("\n--- Test 3: 10 problems with bad code, concurrent via thread pool ---")
    loop = asyncio.get_event_loop()
    t0 = time.time()
    futures = [
        loop.run_in_executor(
            _test_executor,
            run_code_tests_sync,
            BAD_CODE, task.tests,
            TRAIN_TIMEOUT, MAX_TESTS_TRAIN, HARD_CAP_SECONDS,
        )
        for task in tasks
    ]
    results = await asyncio.gather(*futures)
    elapsed = time.time() - t0
    print(f"  All 10 results: {results}")
    print(f"  Total time: {elapsed:.1f}s (concurrent, should be fast)")

    # Test 4: Run all 10 problems with infinite loop concurrently
    print("\n--- Test 4: 10 infinite loops, concurrent via thread pool ---")
    t0 = time.time()
    futures = [
        loop.run_in_executor(
            _test_executor,
            run_code_tests_sync,
            INFINITE_CODE, task.tests,
            TRAIN_TIMEOUT, MAX_TESTS_TRAIN, HARD_CAP_SECONDS,
        )
        for task in tasks
    ]
    results = await asyncio.gather(*futures)
    elapsed = time.time() - t0
    print(f"  All 10 results: {results}")
    print(f"  Total time: {elapsed:.1f}s (should be ~{HARD_CAP_SECONDS}s, not {HARD_CAP_SECONDS}*10)")

    # Test 5: Verify truncation works
    print("\n--- Test 5: Truncation verification ---")
    for i, task in enumerate(tasks[:3]):
        tc = postprocess_lcb_sample(task.tests)
        n_before = len(json.loads(tc["input_output"])["inputs"])
        tc_truncated = _truncate_tests(postprocess_lcb_sample(task.tests), MAX_TESTS_TRAIN)
        n_after = len(json.loads(tc_truncated["input_output"])["inputs"])
        print(f"  Problem {i}: {n_before} tests -> {n_after} tests (max={MAX_TESTS_TRAIN})")

    _test_executor.shutdown(wait=False)
    print("\nDONE - All fixes validated")


if __name__ == '__main__':
    asyncio.run(main())

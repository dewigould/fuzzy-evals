"""Grade generated coding traces by format validation + unit test execution.

Stage 1: Format check — <think>...</think> then ```python``` code block
Stage 2: Unit test execution — write solution.py, run tests in subprocess

Usage:
    python grade_code.py [--test-concurrency N] [--test-timeout N] [--effortful]
"""

import argparse
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

# Import local config BEFORE adding parent to sys.path (parent also has config.py)
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CODE_DATA_DIR, CODE_GENERATIONS_PATH, CODE_GRADED_PATH,
    EFFORTFUL_CODE_DATA_DIR, EFFORTFUL_CODE_GENERATIONS_PATH, EFFORTFUL_CODE_GRADED_PATH,
    THINKING_CODE_DATA_DIR, THINKING_CODE_GENERATIONS_PATH, THINKING_CODE_GRADED_PATH,
)

# Add parent for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import extract_code_from_response


def check_format(generation: str) -> tuple[bool, str]:
    """Check that generation has <think>...</think> then ```python``` code block.

    Returns (valid, reason).
    """
    if "<think>" not in generation:
        return False, "missing <think>"
    if "</think>" not in generation:
        return False, "missing </think>"
    # Reject wrong tag variants
    if "<thinking>" in generation or "</thinking>" in generation:
        return False, "uses <thinking> instead of <think>"

    # Check code block appears after </think>
    think_end = generation.rindex("</think>")
    after_think = generation[think_end + len("</think>"):]
    if "```python" not in after_think and "```\n" not in after_think:
        return False, "no ```python``` block after </think>"

    return True, "ok"


def run_kodcode_tests_sync(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run code against KodCode pytest-style tests in isolated subprocess.

    Returns (passed, detail).
    """
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
            if result.returncode == 0:
                return True, "passed"
            else:
                detail = result.stderr[:500] if result.stderr else "tests failed"
                return False, detail
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, f"error: {e}"


async def main_async(args):
    # Select paths based on flags
    if args.output_dir:
        data_dir = Path(args.output_dir)
        generations_path = data_dir / "generations.jsonl"
        graded_path = data_dir / "graded.jsonl"
        print(f"Grading code traces from {data_dir}")
    elif args.thinking:
        data_dir = THINKING_CODE_DATA_DIR
        generations_path = THINKING_CODE_GENERATIONS_PATH
        graded_path = THINKING_CODE_GRADED_PATH
        print("Grading THINKING model code traces")
    elif args.effortful:
        data_dir = EFFORTFUL_CODE_DATA_DIR
        generations_path = EFFORTFUL_CODE_GENERATIONS_PATH
        graded_path = EFFORTFUL_CODE_GRADED_PATH
        print("Grading EFFORTFUL code traces")
    else:
        data_dir = CODE_DATA_DIR
        generations_path = CODE_GENERATIONS_PATH
        graded_path = CODE_GRADED_PATH

    data_dir.mkdir(parents=True, exist_ok=True)

    if not generations_path.exists():
        print(f"ERROR: {generations_path} not found. Run generate_code.py first.")
        sys.exit(1)

    # Load generations
    generations = []
    with open(generations_path) as f:
        for line in f:
            try:
                generations.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(generations)} generations from {generations_path}")

    # ── Format validation ──────────────────────────────────────────────────
    print("\n── Format check: <think>...</think>...```python``` ──")
    format_fail = 0
    format_ok = []
    results = []

    for gen in generations:
        text = str(gen.get("generation", ""))
        if text.startswith("ERROR:"):
            results.append({**gen, "code_extracted": False, "grading_method": "gen_error",
                            "correct": False, "format_valid": False, "detail": "generation error"})
            format_fail += 1
            continue

        valid, reason = check_format(text)
        if not valid:
            results.append({**gen, "code_extracted": False, "grading_method": f"bad_format:{reason}",
                            "correct": False, "format_valid": False, "detail": reason})
            format_fail += 1
        else:
            format_ok.append(gen)

    print(f"  Format valid: {len(format_ok)}/{len(generations)}")
    if format_fail:
        print(f"  Format rejected: {format_fail}")

    # ── Code extraction ────────────────────────────────────────────────────
    print("\n── Code extraction ──")
    code_ok = []
    no_code = 0
    for gen in format_ok:
        code = extract_code_from_response(gen["generation"])
        if code is None:
            results.append({**gen, "code_extracted": False, "grading_method": "no_code",
                            "correct": False, "format_valid": True, "detail": "no code block found"})
            no_code += 1
        else:
            gen["_extracted_code"] = code
            code_ok.append(gen)

    print(f"  Code extracted: {len(code_ok)}/{len(format_ok)}")
    if no_code:
        print(f"  No code block found: {no_code}")

    # ── Unit test execution ────────────────────────────────────────────────
    print(f"\n── Unit test execution on {len(code_ok)} traces (concurrency={args.test_concurrency}) ──")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.test_concurrency)
    loop = asyncio.get_event_loop()

    t0 = time.time()
    n_correct = 0
    n_failed = 0

    # Process in batches
    batch_size = 200
    for batch_start in range(0, len(code_ok), batch_size):
        batch_end = min(batch_start + batch_size, len(code_ok))
        batch = code_ok[batch_start:batch_end]

        futures = []
        for gen in batch:
            fut = loop.run_in_executor(
                executor,
                run_kodcode_tests_sync,
                gen["_extracted_code"],
                gen["test"],
                args.test_timeout,
            )
            futures.append((gen, fut))

        for gen, fut in futures:
            passed, detail = await fut
            code = gen.pop("_extracted_code")
            rec = {
                **gen,
                "code_extracted": True,
                "extracted_code": code,
                "grading_method": "unit_test",
                "correct": passed,
                "format_valid": True,
                "detail": detail if not passed else "passed",
            }
            results.append(rec)
            if passed:
                n_correct += 1
            else:
                n_failed += 1

        elapsed = time.time() - t0
        total_tested = batch_end
        print(f"  [{total_tested}/{len(code_ok)}] {n_correct} correct, "
              f"{n_failed} failed ({elapsed:.0f}s)")

    executor.shutdown(wait=False)

    # ── Add messages field for training compatibility ────────────────────────
    for rec in results:
        if rec["correct"]:
            rec["messages"] = [
                {"role": "user", "content": rec["question"]},
                {"role": "assistant", "content": rec["generation"]},
            ]

    # ── Write output ────────────────────────────────────────────────────────
    total = len(generations)
    with open(graded_path, "w") as f:
        for rec in results:
            # Remove internal fields
            rec.pop("_extracted_code", None)
            f.write(json.dumps(rec) + "\n")

    total_correct = n_correct
    print(f"\n{'='*60}")
    print(f"FINAL: {total_correct}/{total} correct ({100*total_correct/total:.1f}%)")
    print(f"  Passed tests: {n_correct}  |  Failed tests: {n_failed}  |  "
          f"No code: {no_code}  |  Bad format: {format_fail}")
    print(f"Saved to {graded_path}")

    # Also write correct-only file
    correct_path = data_dir / "correct_only.jsonl"
    n_correct_written = 0
    with open(correct_path, "w") as f:
        for rec in results:
            if rec["correct"]:
                f.write(json.dumps(rec) + "\n")
                n_correct_written += 1
    print(f"Correct-only traces: {correct_path} ({n_correct_written} records)")


def main():
    parser = argparse.ArgumentParser(description="Grade generated coding traces")
    parser.add_argument("--test-concurrency", type=int, default=16,
                        help="Number of parallel test execution workers")
    parser.add_argument("--test-timeout", type=int, default=10,
                        help="Timeout per test in seconds")
    parser.add_argument("--effortful", action="store_true",
                        help="Grade effortful traces from data_code_effortful/")
    parser.add_argument("--thinking", action="store_true",
                        help="Grade thinking model traces from data_code_thinking/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override data directory (must contain generations.jsonl)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

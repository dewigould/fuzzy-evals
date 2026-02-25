"""
Quick eval: Test Llama-3.1-8B-Instruct on 50 easy KodCode problems
to see if the base model can solve any (needed for RLVR signal).
"""
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import concurrent.futures

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from datasets import load_dataset
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Reuse extract_code from existing code RLVR
from run_code_rlvr import extract_code

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_PROBLEMS = 50
N_SAMPLES = 4

_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def run_kodcode_tests_sync(code, test_code, timeout=10):
    """Run code against KodCode pytest-style tests.

    Writes the model's code as solution.py, the test as test_solution.py,
    and runs pytest in a subprocess with resource limits.
    """
    resource_wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
    )

    # Build a runner script that:
    # 1. Sets resource limits
    # 2. Imports the test module and runs all test functions
    runner = resource_wrapper + '''
import sys
import importlib.util
import traceback

# Add tmpdir to path so "from solution import ..." works
sys.path.insert(0, ".")

try:
    # Import solution first to check it loads
    import solution

    # Load test module
    spec = importlib.util.spec_from_file_location("test_solution", "test_solution.py")
    test_mod = importlib.util.module_from_spec(spec)

    # Inject solution functions into test module's namespace
    # (for tests that don't use "from solution import" but call functions directly)
    for name in dir(solution):
        if not name.startswith('_'):
            setattr(test_mod, name, getattr(solution, name))

    spec.loader.exec_module(test_mod)

    # Run all test functions
    test_funcs = [name for name in dir(test_mod) if name.startswith('test_')]
    failed = 0
    for name in test_funcs:
        try:
            getattr(test_mod, name)()
        except Exception as e:
            failed += 1

    if failed > 0:
        sys.exit(1)
    elif len(test_funcs) == 0:
        sys.exit(1)  # No tests found
    else:
        sys.exit(0)
except Exception as e:
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
                cwd=tmpdir,
                capture_output=True,
                timeout=timeout + 2,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def load_kodcode_easy(n=50, seed=42):
    """Load n easy KodCode problems with from-solution-import tests."""
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    # Filter: easy problems with from-solution-import tests
    problems = []
    for row in ds:
        if row['gpt_difficulty'] != 'easy':
            continue
        if 'from solution import' not in row['test']:
            continue
        # Skip problems needing numpy/pandas/torch (may cause issues)
        test_lower = row['test'].lower()
        if any(lib in test_lower for lib in ['import numpy', 'import pandas', 'import torch']):
            continue
        problems.append(row)

    rng = random.Random(seed)
    rng.shuffle(problems)
    return problems[:n]


async def main():
    print("Loading KodCode easy problems...")
    problems = load_kodcode_easy(N_PROBLEMS)
    print(f"Loaded {len(problems)} easy problems")

    # Show some examples
    for i in range(min(3, len(problems))):
        p = problems[i]
        print(f"\n  Problem {i}: {p['question'][:100]}...")
        print(f"    gpt_pass: {p['gpt_pass_percentage']}, tests start: {p['test'][:60]}...")

    # First, verify test runner works with reference solutions
    print("\n--- Verifying test runner with reference solutions ---")
    n_ref_pass = 0
    for i in range(min(10, len(problems))):
        p = problems[i]
        ref_code = p['solution']
        passed = run_kodcode_tests_sync(ref_code, p['test'])
        status = "PASS" if passed else "FAIL"
        if passed:
            n_ref_pass += 1
        print(f"  Ref solution {i}: {status}")
    print(f"  Reference solutions: {n_ref_pass}/{min(10, len(problems))} pass")

    if n_ref_pass < 8:
        print("WARNING: Reference solutions failing too often, test runner may have issues")

    # Now eval the 8B model
    print(f"\n{'='*60}")
    print(f"Evaluating {MODEL_NAME} on {len(problems)} easy KodCode problems")
    print(f"  {N_SAMPLES} samples per problem")
    print(f"{'='*60}")

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(base_model=MODEL_NAME)

    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(5)

    results = []
    n_solved_any = 0
    n_solved_total = 0

    async def eval_one(idx, problem):
        nonlocal n_solved_any, n_solved_total

        prompt = (
            "You are an expert Python programmer. Solve the following problem.\n\n"
            + problem['question']
            + "\n\nProvide your solution in a ```python code block."
        )
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)

        async with semaphore:
            try:
                resp = await client.sample_async(
                    model_input, num_samples=N_SAMPLES,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=2048, temperature=0.7,
                        stop=renderer.get_stop_sequences(),
                    ),
                )
            except Exception as e:
                print(f"  Problem {idx}: ERROR: {e}")
                return

        any_correct = False
        n_correct = 0
        for seq in resp.sequences:
            parsed, _ = renderer.parse_response(seq.tokens)
            text = parsed['content']
            code = extract_code(text)
            if code is None:
                continue

            passed = await loop.run_in_executor(
                _test_executor,
                run_kodcode_tests_sync, code, problem['test'],
            )
            if passed:
                n_correct += 1
                any_correct = True

        if any_correct:
            n_solved_any += 1
        n_solved_total += n_correct

        status = f"PASS ({n_correct}/{N_SAMPLES})" if any_correct else "FAIL"
        print(f"  Problem {idx} (gpt_pass={problem['gpt_pass_percentage']:.1f}): {status}")
        results.append({
            'idx': idx,
            'any_correct': any_correct,
            'n_correct': n_correct,
            'gpt_pass_pct': problem['gpt_pass_percentage'],
        })

    t0 = time.time()
    for batch_start in range(0, len(problems), 10):
        batch_end = min(batch_start + 10, len(problems))
        await asyncio.gather(*[eval_one(i, problems[i]) for i in range(batch_start, batch_end)])
        elapsed = time.time() - t0
        done = len(results)
        solved = sum(1 for r in results if r['any_correct'])
        print(f"\n  Progress: {done}/{len(problems)} | "
              f"Solved: {solved}/{done} ({solved/max(done,1)*100:.1f}%) | "
              f"{elapsed:.0f}s\n")

    elapsed = time.time() - t0
    pass_at_1 = n_solved_total / (len(problems) * N_SAMPLES) if problems else 0
    pass_at_k = n_solved_any / len(problems) if problems else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {MODEL_NAME} on {len(problems)} easy KodCode problems")
    print(f"  pass@1 (approx): {pass_at_1*100:.1f}%")
    print(f"  pass@{N_SAMPLES}: {pass_at_k*100:.1f}% ({n_solved_any}/{len(problems)})")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    # Save results
    out = {
        'model': MODEL_NAME,
        'dataset': 'KodCode-Light-RL-10K',
        'filter': 'easy + from-solution-import',
        'n_problems': len(problems),
        'n_samples': N_SAMPLES,
        'pass_at_1': pass_at_1,
        f'pass_at_{N_SAMPLES}': pass_at_k,
        'n_solved_any': n_solved_any,
        'n_solved_total': n_solved_total,
        'time_seconds': elapsed,
        'details': results,
    }
    out_path = '/workspace/results_10_02/code_rlvr_eval/kodcode_8b_quick_eval.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    asyncio.run(main())

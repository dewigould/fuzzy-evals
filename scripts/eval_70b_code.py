"""
Quick evaluation: Can Llama-3.3-70B-Instruct solve DeepCoder problems?
Tests 50 problems with the base 70B model to estimate solve rate.
"""
import asyncio
import json
import sys
import time

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from run_code_rlvr import (
    extract_code, run_code_tests_sync, HARD_CAP_SECONDS, TRAIN_TIMEOUT,
    _test_executor, load_deepcoder_tasks_streaming,
)

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
N_PROBLEMS = 50
N_SAMPLES = 4  # samples per problem to estimate pass@1 and pass@4


async def main():
    print("=" * 60)
    print(f"Evaluating {MODEL_NAME} on {N_PROBLEMS} DeepCoder problems")
    print(f"  {N_SAMPLES} samples per problem")
    print(f"  HARD_CAP={HARD_CAP_SECONDS}s, TIMEOUT={TRAIN_TIMEOUT}s")
    print("=" * 60)

    # Set up model
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(base_model=MODEL_NAME)

    # Load problems
    print("\nLoading problems...")
    tasks = load_deepcoder_tasks_streaming("train", seed=42)
    tasks = tasks[:N_PROBLEMS]
    print(f"  Loaded {len(tasks)} problems")

    # Generate and test
    results = []
    n_solved_any = 0  # problems with at least 1 correct sample
    n_solved_all = 0  # total correct samples
    total_samples = 0

    semaphore = asyncio.Semaphore(5)
    loop = asyncio.get_event_loop()

    async def eval_one(idx, task):
        nonlocal n_solved_any, n_solved_all, total_samples

        question = task.problem + "\n\nProvide your solution in a ```python code block."
        messages = [{"role": "user", "content": question}]
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
                print(f"  Problem {idx}: ERROR generating: {e}")
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
                run_code_tests_sync, code, task.tests,
                TRAIN_TIMEOUT, HARD_CAP_SECONDS,
            )
            if passed:
                n_correct += 1
                any_correct = True

        total_samples += N_SAMPLES
        if any_correct:
            n_solved_any += 1
        n_solved_all += n_correct

        status = f"pass@{N_SAMPLES}=YES ({n_correct}/{N_SAMPLES})" if any_correct else f"FAIL (0/{N_SAMPLES})"
        print(f"  Problem {idx}: {status}")
        results.append({
            'idx': idx,
            'any_correct': any_correct,
            'n_correct': n_correct,
            'n_samples': N_SAMPLES,
        })

    # Process in batches of 10
    t0 = time.time()
    for batch_start in range(0, N_PROBLEMS, 10):
        batch_end = min(batch_start + 10, N_PROBLEMS)
        batch_tasks = [eval_one(i, tasks[i]) for i in range(batch_start, batch_end)]
        await asyncio.gather(*batch_tasks)
        elapsed = time.time() - t0
        print(f"\n  --- Progress: {batch_end}/{N_PROBLEMS} | "
              f"Solved: {n_solved_any}/{batch_end} ({n_solved_any/batch_end*100:.1f}%) | "
              f"Time: {elapsed:.0f}s ---\n")

    elapsed = time.time() - t0

    # Summary
    pass_at_1 = n_solved_all / total_samples if total_samples else 0
    pass_at_k = n_solved_any / N_PROBLEMS if N_PROBLEMS else 0

    print("\n" + "=" * 60)
    print(f"RESULTS: {MODEL_NAME} on {N_PROBLEMS} DeepCoder problems")
    print(f"  pass@1 (approx): {pass_at_1*100:.1f}% ({n_solved_all}/{total_samples})")
    print(f"  pass@{N_SAMPLES}: {pass_at_k*100:.1f}% ({n_solved_any}/{N_PROBLEMS})")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 60)

    # Save results
    summary = {
        'model': MODEL_NAME,
        'n_problems': N_PROBLEMS,
        'n_samples': N_SAMPLES,
        'pass_at_1': pass_at_1,
        f'pass_at_{N_SAMPLES}': pass_at_k,
        'n_solved_any': n_solved_any,
        'n_solved_all': n_solved_all,
        'total_samples': total_samples,
        'time_seconds': elapsed,
        'details': results,
    }
    out_path = '/workspace/results_10_02/code_rlvr_eval/70b_deepcoder_eval.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    asyncio.run(main())

"""
Evaluate all 3 models (base, math RLVR, code RLVR) on 200 held-out easy
KodCode problems to add a KodCode bar to the comparison chart.
"""
import asyncio
import concurrent.futures
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from run_code_rlvr import extract_code, load_kodcode_tasks, run_kodcode_tests_sync

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_TEST = 200
N_SAMPLES = 1  # pass@1 for fair comparison with other benchmarks

# Checkpoint paths (from sweep results)
MATH_RLVR_CKPT = "tinker://c83350ad-1856-520c-998f-7dadd998f07d:train:0/sampler_weights/000050"
CODE_RLVR_CKPT = "tinker://5af2a568-2f98-57f6-b4da-ae0b4e880805:train:0/sampler_weights/000050"

_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


async def eval_model(label, sampling_client, renderer, problems):
    """Evaluate a single model on problems."""
    print(f"\n{'='*60}")
    print(f"Evaluating {label} on {len(problems)} KodCode problems")
    print(f"{'='*60}")

    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(5)
    results = []
    n_correct = 0
    completions = []

    async def eval_one(idx, problem):
        nonlocal n_correct

        prompt = (
            "You are an expert Python programmer. Solve the following problem.\n\n"
            + problem.question
            + "\n\nProvide your solution in a ```python code block."
        )
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)

        async with semaphore:
            try:
                resp = await sampling_client.sample_async(
                    model_input, num_samples=N_SAMPLES,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=1024, temperature=0.0,  # greedy for pass@1
                        stop=renderer.get_stop_sequences(),
                    ),
                )
            except Exception as e:
                print(f"  [{label}] Problem {idx}: ERROR: {e}")
                return

        parsed, _ = renderer.parse_response(resp.sequences[0].tokens)
        text = parsed['content']
        code = extract_code(text)

        passed = False
        if code is not None:
            passed = await loop.run_in_executor(
                _test_executor,
                run_kodcode_tests_sync, code, problem.test_code,
            )

        if passed:
            n_correct += 1

        results.append({'idx': idx, 'correct': passed})
        completions.append({'idx': idx, 'text': text, 'correct': passed})

    t0 = time.time()
    for batch_start in range(0, len(problems), 20):
        batch_end = min(batch_start + 20, len(problems))
        await asyncio.gather(*[eval_one(i, problems[i]) for i in range(batch_start, batch_end)])
        done = len(results)
        correct_so_far = sum(1 for r in results if r['correct'])
        elapsed = time.time() - t0
        print(f"  [{label}] {done}/{len(problems)} | "
              f"{correct_so_far}/{done} correct ({correct_so_far/max(done,1)*100:.1f}%) | "
              f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    accuracy = n_correct / len(problems)
    print(f"\n  [{label}] RESULT: {n_correct}/{len(problems)} ({accuracy*100:.1f}%) in {elapsed:.0f}s")

    return {
        'label': label,
        'n_problems': len(problems),
        'n_correct': n_correct,
        'accuracy': accuracy,
        'time_seconds': elapsed,
        'completions': completions,
    }


async def main():
    # Load held-out test problems
    print("Loading KodCode easy problems...")
    all_tasks = load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42)
    test_problems = all_tasks[-N_TEST:]
    print(f"Using {len(test_problems)} held-out test problems (indices {len(all_tasks)-N_TEST}+)")

    # Set up renderer (shared across all models)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()

    # 1. Base model
    base_client = sc.create_sampling_client(base_model=MODEL_NAME)
    base_results = await eval_model("base", base_client, renderer, test_problems)

    # 2. Math RLVR
    math_client = sc.create_sampling_client(model_path=MATH_RLVR_CKPT)
    print(f"\nLoaded math RLVR from {MATH_RLVR_CKPT}")
    math_results = await eval_model("math_rlvr", math_client, renderer, test_problems)

    # 3. Code RLVR
    code_client = sc.create_sampling_client(model_path=CODE_RLVR_CKPT)
    print(f"\nLoaded code RLVR from {CODE_RLVR_CKPT}")
    code_results = await eval_model("code_rlvr", code_client, renderer, test_problems)

    # Summary
    print(f"\n{'='*60}")
    print(f"KODCODE-200 RESULTS (held-out easy problems, pass@1)")
    print(f"{'='*60}")
    print(f"  Base:      {base_results['accuracy']*100:.1f}% ({base_results['n_correct']}/{N_TEST})")
    print(f"  Math RLVR: {math_results['accuracy']*100:.1f}% ({math_results['n_correct']}/{N_TEST})")
    print(f"  Code RLVR: {code_results['accuracy']*100:.1f}% ({code_results['n_correct']}/{N_TEST})")

    # Save results
    out = {
        'n_problems': N_TEST,
        'base': {k: v for k, v in base_results.items() if k != 'completions'},
        'math_rlvr': {k: v for k, v in math_results.items() if k != 'completions'},
        'code_rlvr': {k: v for k, v in code_results.items() if k != 'completions'},
    }
    out_path = '/workspace/results_10_02/code_rlvr_eval/kodcode_200_3model.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save completions for response length analysis
    completions_path = '/workspace/results_10_02/code_rlvr_eval/kodcode_200_completions.json'
    with open(completions_path, 'w') as f:
        json.dump({
            'base': base_results['completions'],
            'math_rlvr': math_results['completions'],
            'code_rlvr': code_results['completions'],
        }, f, indent=2)
    print(f"Saved completions to {completions_path}")


if __name__ == '__main__':
    asyncio.run(main())

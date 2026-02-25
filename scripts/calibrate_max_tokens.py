"""
Step 0: Calibrate max_tokens for RLVR training.
Samples base Llama-3.1-8B-Instruct on 200 random Hendrycks MATH training problems
with max_tokens=4096, measures actual output token lengths, and recommends a
max_tokens value for training.
"""

import asyncio
import json
import math
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    MathEnv,
    _get_hendrycks_math_train,
    extract_boxed,
    safe_grade,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_SAMPLES = 200
CALIBRATION_MAX_TOKENS = 4096
RESULTS_FILE = Path('/workspace/results_10_02/calibration.json')


async def main():
    # Setup
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    convo_prefix = MathEnv.standard_fewshot_prefix()

    # Create sampling client for base model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Load training problems and sample 200 randomly
    print(f"Loading Hendrycks MATH training set...")
    train_ds = _get_hendrycks_math_train()
    print(f"  {len(train_ds)} training problems loaded")

    random.seed(42)
    indices = random.sample(range(len(train_ds)), min(N_SAMPLES, len(train_ds)))

    # Sample responses
    print(f"\nSampling {N_SAMPLES} problems with max_tokens={CALIBRATION_MAX_TOKENS}...")
    results = []
    semaphore = asyncio.Semaphore(20)  # Limit concurrency

    async def sample_one(idx):
        row = train_ds[idx]
        problem = row['problem']
        solution = row['solution']

        # Extract ground truth answer
        try:
            ground_truth = extract_boxed(solution)
        except ValueError:
            return None

        # Build prompt exactly matching training format
        question = problem + MathEnv.question_suffix()
        messages = convo_prefix + [{"role": "user", "content": question}]
        model_input = renderer.build_generation_prompt(messages)

        async with semaphore:
            response = await sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=CALIBRATION_MAX_TOKENS,
                    temperature=1.0,
                    stop=renderer.get_stop_sequences(),
                ),
            )

        tokens = response.sequences[0].tokens
        token_count = len(tokens)

        # Decode and check correctness
        parsed_msg, _ = renderer.parse_response(tokens)
        text = parsed_msg['content']

        try:
            predicted = extract_boxed(text)
            correct = safe_grade(predicted, ground_truth)
        except (ValueError, Exception):
            correct = False

        has_boxed = '\\boxed{' in text or '\\boxed ' in text

        return {
            'idx': idx,
            'token_count': token_count,
            'correct': correct,
            'has_boxed': has_boxed,
            'hit_limit': token_count >= CALIBRATION_MAX_TOKENS - 1,
        }

    tasks = [sample_one(idx) for idx in indices]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS ({len(results)} problems sampled)")
    print(f"{'='*60}")

    # Compute stats
    all_lengths = [r['token_count'] for r in results]
    correct_lengths = [r['token_count'] for r in results if r['correct']]
    wrong_lengths = [r['token_count'] for r in results if not r['correct']]
    hit_limit = sum(1 for r in results if r['hit_limit'])
    no_boxed = sum(1 for r in results if not r['has_boxed'])

    def print_percentiles(name, lengths):
        if not lengths:
            print(f"  {name}: no data")
            return
        percentiles = [50, 75, 90, 95, 99]
        print(f"  {name} (n={len(lengths)}):")
        print(f"    Mean: {np.mean(lengths):.0f} tokens")
        for p in percentiles:
            val = np.percentile(lengths, p)
            print(f"    P{p}: {val:.0f} tokens")
        print(f"    Max: {max(lengths)} tokens")

    print(f"\nCorrect: {len(correct_lengths)}/{len(results)} ({100*len(correct_lengths)/len(results):.1f}%)")
    print(f"Hit max_tokens limit: {hit_limit}/{len(results)}")
    print(f"No \\boxed{{}} in output: {no_boxed}/{len(results)}")

    print_percentiles("\nAll responses", all_lengths)
    print_percentiles("\nCorrect responses", correct_lengths)
    print_percentiles("\nWrong responses", wrong_lengths)

    # Recommend max_tokens
    if correct_lengths:
        p99_correct = np.percentile(correct_lengths, 99)
        # Round up to nearest power of 2 with 20% margin
        recommended = p99_correct * 1.2
        power_of_2 = 2 ** math.ceil(math.log2(recommended))
        # Clamp to reasonable range
        recommended_final = min(max(power_of_2, 1024), 4096)
    else:
        recommended_final = 2048

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: max_tokens = {int(recommended_final)}")
    if correct_lengths:
        print(f"  (P99 of correct = {p99_correct:.0f}, x1.2 = {recommended:.0f}, rounded to power of 2)")
    print(f"{'='*60}")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    calibration_data = {
        'n_sampled': len(results),
        'n_correct': len(correct_lengths),
        'recommended_max_tokens': int(recommended_final),
        'percentiles_all': {str(p): float(np.percentile(all_lengths, p)) for p in [50, 75, 90, 95, 99]},
        'percentiles_correct': {str(p): float(np.percentile(correct_lengths, p)) for p in [50, 75, 90, 95, 99]} if correct_lengths else {},
        'max_all': int(max(all_lengths)),
        'max_correct': int(max(correct_lengths)) if correct_lengths else 0,
        'hit_limit_count': hit_limit,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"\nSaved calibration data to {RESULTS_FILE}")

    return int(recommended_final)


if __name__ == '__main__':
    result = asyncio.run(main())
    print(f"\n{result}")

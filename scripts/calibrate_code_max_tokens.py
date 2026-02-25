"""
Step 0: Calibrate max_tokens for Code RLVR training with KodCode.
Samples base Llama-3.1-8B-Instruct on 200 random easy KodCode problems
with max_tokens=4096, measures actual output token lengths, and recommends a
max_tokens value for training.

Uses "has code block" as proxy for a reasonable response (skips expensive
subprocess-based test execution to keep calibration fast).
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
from tinker_cookbook.tokenizer_utils import get_tokenizer

from run_code_rlvr import extract_code, load_kodcode_tasks

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_SAMPLES = 200
CALIBRATION_MAX_TOKENS = 4096
RESULTS_FILE = Path('/workspace/results_10_02/code_calibration_kodcode.json')


async def main():
    # Setup
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Create sampling client for base model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Load easy KodCode tasks and sample 200
    all_tasks = load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42)

    random.seed(42)
    indices = random.sample(range(len(all_tasks)), min(N_SAMPLES, len(all_tasks)))

    print(f"\nSampling {N_SAMPLES} problems with max_tokens={CALIBRATION_MAX_TOKENS}...")
    print(f"  Processing in batches of 10 with concurrency=5...")
    results = []

    async def sample_one(idx, semaphore):
        task = all_tasks[idx]
        question = (task.question +
                    "\n\nProvide your solution in a ```python code block.")
        messages = [{"role": "user", "content": question}]
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
        parsed_msg, _ = renderer.parse_response(tokens)
        text = parsed_msg['content']
        has_code = extract_code(text) is not None

        return {
            'idx': idx,
            'token_count': token_count,
            'has_code': has_code,
            'hit_limit': token_count >= CALIBRATION_MAX_TOKENS - 1,
        }

    BATCH_SIZE = 10
    semaphore = asyncio.Semaphore(5)
    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
        batch_tasks = [sample_one(idx, semaphore) for idx in batch_indices]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend([r for r in batch_results if r is not None])
        n_with_code = sum(1 for r in results if r['has_code'])
        print(f"  [{len(results)}/{len(indices)}] done — {n_with_code} with code blocks")

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS ({len(results)} problems sampled)")
    print(f"{'='*60}")

    all_lengths = [r['token_count'] for r in results]
    code_lengths = [r['token_count'] for r in results if r['has_code']]
    hit_limit = sum(1 for r in results if r['hit_limit'])
    no_code = sum(1 for r in results if not r['has_code'])

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

    print(f"\nWith code block: {len(code_lengths)}/{len(results)} ({100*len(code_lengths)/len(results):.1f}%)")
    print(f"Hit max_tokens limit: {hit_limit}/{len(results)}")
    print(f"No code block in output: {no_code}/{len(results)}")

    print_percentiles("\nAll responses", all_lengths)
    print_percentiles("\nResponses with code", code_lengths)

    # Recommend max_tokens
    if code_lengths:
        p99 = np.percentile(code_lengths, 99)
        recommended = p99 * 1.2
        power_of_2 = 2 ** math.ceil(math.log2(recommended))
        recommended_final = min(max(power_of_2, 1024), 8192)
    else:
        p99 = None
        recommended_final = 2048

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: max_tokens = {int(recommended_final)}")
    if p99 is not None:
        print(f"  (P99 of code responses = {p99:.0f}, x1.2 = {recommended:.0f}, rounded to power of 2)")
    print(f"{'='*60}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    calibration_data = {
        'dataset': 'KodCode-Light-RL-10K',
        'filter': 'easy + from-solution-import',
        'n_sampled': len(results),
        'n_with_code': len(code_lengths),
        'recommended_max_tokens': int(recommended_final),
        'percentiles_all': {str(p): float(np.percentile(all_lengths, p)) for p in [50, 75, 90, 95, 99]},
        'percentiles_code': {str(p): float(np.percentile(code_lengths, p)) for p in [50, 75, 90, 95, 99]} if code_lengths else {},
        'max_all': int(max(all_lengths)),
        'max_code': int(max(code_lengths)) if code_lengths else 0,
        'hit_limit_count': hit_limit,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"\nSaved calibration data to {RESULTS_FILE}")

    return int(recommended_final)


if __name__ == '__main__':
    result = asyncio.run(main())
    print(f"\n{result}")

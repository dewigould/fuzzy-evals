"""
Quick eval: test Llama-3.3-70B-Instruct AND Llama-3.1-8B-Instruct on 50 EASY
TACO problems to see if either model can solve them.
"""
import asyncio
import json
import random
import sys
import time

import pandas as pd
from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from run_code_rlvr import (
    extract_code, run_code_tests_sync, HARD_CAP_SECONDS, TRAIN_TIMEOUT,
    _test_executor,
)
from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample

N_PROBLEMS = 50
N_SAMPLES = 4


def load_taco_easy(n=50, seed=42):
    """Load n EASY problems from TACO."""
    # Load first shard (has 1005 EASY problems)
    url = 'https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00000-of-00009.parquet'
    df = pd.read_parquet(url)
    easy = df[df['difficulty'] == 'EASY'].reset_index(drop=True)

    # Sample n problems
    rng = random.Random(seed)
    indices = rng.sample(range(len(easy)), min(n, len(easy)))

    problems = []
    for idx in indices:
        row = easy.iloc[idx]
        question = row['question']
        io_raw = row.get('input_output', '')

        if not question or not io_raw:
            continue

        # Parse input_output to get test cases
        try:
            io = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
            inputs = io.get('inputs', [])
            outputs = io.get('outputs', [])
            if not inputs or not outputs:
                continue
        except:
            continue

        # Convert to the format our test runner expects
        test_cases = []
        fn_name = io.get('fn_name', None)
        for inp, out in zip(inputs, outputs):
            tc = {'input': inp, 'output': out, 'metadata': {}}
            if fn_name:
                tc['testtype'] = 'functional'
                tc['metadata']['func_name'] = fn_name
            else:
                tc['testtype'] = 'stdin_stdout'
            test_cases.append(tc)

        problems.append({
            'question': question,
            'tests': test_cases,
            'n_tests': len(test_cases),
            'source': row.get('source', ''),
            'name': row.get('name', ''),
        })

    return problems


async def eval_model(model_name, problems, label):
    """Evaluate a single model on the problems."""
    print(f"\n{'='*60}")
    print(f"Evaluating {label}: {model_name}")
    print(f"{'='*60}")

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(base_model=model_name)

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
                print(f"  [{label}] Problem {idx}: ERROR: {e}")
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
                run_code_tests_sync, code, problem['tests'],
                TRAIN_TIMEOUT, HARD_CAP_SECONDS,
            )
            if passed:
                n_correct += 1
                any_correct = True

        if any_correct:
            n_solved_any += 1
        n_solved_total += n_correct

        status = f"PASS ({n_correct}/{N_SAMPLES})" if any_correct else f"FAIL"
        print(f"  [{label}] Problem {idx} ({problem['n_tests']} tests): {status}")
        results.append({
            'idx': idx,
            'any_correct': any_correct,
            'n_correct': n_correct,
        })

    t0 = time.time()
    for batch_start in range(0, len(problems), 10):
        batch_end = min(batch_start + 10, len(problems))
        await asyncio.gather(*[eval_one(i, problems[i]) for i in range(batch_start, batch_end)])
        elapsed = time.time() - t0
        solved_so_far = sum(1 for r in results if r['any_correct'])
        done = len(results)
        print(f"  [{label}] Progress: {done}/{len(problems)} | "
              f"Solved: {solved_so_far}/{done} ({solved_so_far/max(done,1)*100:.1f}%) | "
              f"{elapsed:.0f}s\n")

    elapsed = time.time() - t0
    pass_at_1 = n_solved_total / (len(problems) * N_SAMPLES) if problems else 0
    pass_at_k = n_solved_any / len(problems) if problems else 0

    print(f"\n  [{label}] RESULTS:")
    print(f"    pass@1 (approx): {pass_at_1*100:.1f}%")
    print(f"    pass@{N_SAMPLES}: {pass_at_k*100:.1f}% ({n_solved_any}/{len(problems)})")
    print(f"    Time: {elapsed:.0f}s")

    return {
        'model': model_name,
        'label': label,
        'pass_at_1': pass_at_1,
        f'pass_at_{N_SAMPLES}': pass_at_k,
        'n_solved_any': n_solved_any,
        'n_solved_total': n_solved_total,
        'n_problems': len(problems),
        'time_seconds': elapsed,
    }


async def main():
    print("Loading 50 EASY TACO problems...")
    problems = load_taco_easy(N_PROBLEMS)
    print(f"Loaded {len(problems)} problems")

    # Show test count distribution
    test_counts = [p['n_tests'] for p in problems]
    print(f"Test counts: min={min(test_counts)}, max={max(test_counts)}, "
          f"median={sorted(test_counts)[len(test_counts)//2]}")

    # Evaluate both models
    results_8b = await eval_model(
        "meta-llama/Llama-3.1-8B-Instruct", problems, "8B")
    results_70b = await eval_model(
        "meta-llama/Llama-3.3-70B-Instruct", problems, "70B")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON: TACO EASY (50 problems)")
    print("=" * 60)
    print(f"  8B  pass@1={results_8b['pass_at_1']*100:.1f}%, "
          f"pass@{N_SAMPLES}={results_8b[f'pass_at_{N_SAMPLES}']*100:.1f}%")
    print(f"  70B pass@1={results_70b['pass_at_1']*100:.1f}%, "
          f"pass@{N_SAMPLES}={results_70b[f'pass_at_{N_SAMPLES}']*100:.1f}%")

    out = {'problems': len(problems), '8b': results_8b, '70b': results_70b}
    out_path = '/workspace/results_10_02/code_rlvr_eval/taco_easy_eval.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    asyncio.run(main())

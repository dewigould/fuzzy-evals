"""
Generate formatting examples for the formatting SFT step.
Calls Sonnet 3.5 via OpenRouter to produce 50 math + 50 code CoT examples.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv('/workspace/.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
SONNET_MODEL = 'anthropic/claude-3.5-sonnet'

RESULTS_DIR = Path('/workspace/results_06_02_v2')
DATA_DIR = RESULTS_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


async def call_openrouter(session, messages, temperature=0.3, max_tokens=2048):
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }
    body = {
        'model': SONNET_MODEL,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    retry_delays = [2, 5, 15, 30, 60]
    for delay in retry_delays + [60]:
        try:
            async with session.post(OPENROUTER_URL, headers=headers,
                                   json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            elif 'error' in data:
                err = data['error']
                err_msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                if 'rate' in err_msg.lower() or '429' in err_msg:
                    await asyncio.sleep(delay)
                    continue
                return f"ERROR: {err_msg}"
        except Exception:
            await asyncio.sleep(delay)
    return "ERROR: max retries"


async def main():
    from datasets import load_dataset

    # ── Load math problems (from Hendrycks MATH train, not in MATH-500) ──
    print("Loading Hendrycks MATH algebra problems...")
    math_ds = load_dataset("EleutherAI/hendrycks_math", name="algebra", split="train")
    # Load MATH-500 to exclude
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math500_problems = {p['problem'] for p in math500}
    # Pick 50 problems not in MATH-500
    math_problems = [p for p in math_ds if p['problem'] not in math500_problems][:50]
    print(f"  Selected {len(math_problems)} math problems")

    # ── Load code tasks (BigCodeBench 500-549, not used in eval) ──
    print("Loading BigCodeBench tasks 500-549...")
    code_ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    code_tasks = [code_ds[i] for i in range(500, 550)]
    print(f"  Selected {len(code_tasks)} code tasks")

    all_examples = []
    semaphore = asyncio.Semaphore(5)  # conservative rate limiting

    async with aiohttp.ClientSession() as session:
        # ── Generate math formatting examples ──
        print("Generating math formatting examples...")
        math_tasks = []
        for prob in math_problems:
            user_prompt = (
                "Solve this math problem step by step. Show clear reasoning, "
                "then put your final answer in \\boxed{}.\n\n"
                f"Problem: {prob['problem']}"
            )
            eval_user = f"{prob['problem']}\n\nWrite your answer in \\boxed{{}} format."
            math_tasks.append((user_prompt, eval_user))

        for i, (gen_prompt, eval_user) in enumerate(math_tasks):
            async with semaphore:
                messages = [{'role': 'user', 'content': gen_prompt}]
                resp = await call_openrouter(session, messages)
                if not resp.startswith("ERROR:"):
                    all_examples.append({
                        'messages': [
                            {'role': 'user', 'content': eval_user},
                            {'role': 'assistant', 'content': resp},
                        ]
                    })
                if (i + 1) % 10 == 0:
                    print(f"  Math: {i+1}/50")

        print(f"  Math formatting examples: {sum(1 for e in all_examples)} generated")

        # ── Generate code formatting examples ──
        print("Generating code formatting examples...")
        n_before = len(all_examples)
        code_gen_tasks = []
        for task in code_tasks:
            gen_prompt = (
                "Write a Python solution for the following task. Think through the approach "
                "step by step, then provide clean, well-documented code in a ```python code block.\n\n"
                f"Task:\n{task['instruct_prompt']}"
            )
            code_gen_tasks.append((gen_prompt, task['instruct_prompt']))

        for i, (gen_prompt, eval_user) in enumerate(code_gen_tasks):
            async with semaphore:
                messages = [{'role': 'user', 'content': gen_prompt}]
                resp = await call_openrouter(session, messages)
                if not resp.startswith("ERROR:"):
                    all_examples.append({
                        'messages': [
                            {'role': 'user', 'content': eval_user},
                            {'role': 'assistant', 'content': resp},
                        ]
                    })
                if (i + 1) % 10 == 0:
                    print(f"  Code: {i+1}/50")

        n_code = len(all_examples) - n_before
        print(f"  Code formatting examples: {n_code} generated")

    # ── Save ──
    output_path = DATA_DIR / 'formatting_combined.jsonl'
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"\nSaved {len(all_examples)} formatting examples to {output_path}")


if __name__ == '__main__':
    asyncio.run(main())

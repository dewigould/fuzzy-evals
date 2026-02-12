#!/usr/bin/env python3
"""Debug script to test distilled model responses with different settings.

Tests:
1. Fix the parse_response list bug by handling list content
2. Try with think_prefix=False (no double <think>)
3. Try with higher max_tokens
4. Try different checkpoints (earlier steps)
"""

import asyncio
import json
import sys
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import MODEL_NAME, MAX_TOKENS

TRAINING_BASE = Path(__file__).parent / "training_runs"

# A few easy MATH problems for testing
TEST_PROBLEMS = [
    {
        "question": "Convert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r,\\theta), where r > 0 and 0 \\le \\theta < 2 \\pi.",
        "answer": "(3,\\frac{\\pi}{2})",
    },
    {
        "question": "What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?",
        "answer": "2220",
    },
    {
        "question": "If $2^{10} \\cdot 2^{15}$ is expressed as some integer to the fifth power, what is that integer?",
        "answer": "32",
    },
]

MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question, "
    "putting your final answer inside \\boxed{}"
)

CODE_TEST_PROBLEMS = [
    {
        "question": "Write a Python function called `reverse_string` that takes a string and returns it reversed.",
        "answer": "def reverse_string(s): return s[::-1]",
    },
    {
        "question": "Write a Python function called `is_palindrome` that takes a string and returns True if it is a palindrome, False otherwise.",
        "answer": "def is_palindrome(s): return s == s[::-1]",
    },
]

CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question. "
    "Enclose your code within delimiters as follows: "
    "```python \n #YOUR CODE HERE \n ``` \n\n"
)


def extract_content_as_string(content) -> str:
    """Handle both str and list[ContentPart] from parse_response."""
    if isinstance(content, str):
        return content
    # It's a list of content parts
    parts = []
    for part in content:
        if hasattr(part, "thinking"):
            parts.append(f"<think>{part.thinking}</think>")
        elif hasattr(part, "text"):
            parts.append(part.text)
        else:
            parts.append(str(part))
    return "".join(parts)


async def generate_fixed(
    sampling_client, renderer, tokenizer, messages,
    max_tokens=MAX_TOKENS, think_prefix=True,
):
    """Generate with fix for the list content bug."""
    model_input = renderer.build_generation_prompt(messages)

    if think_prefix and tokenizer is not None:
        prefix_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)
        for token_id in prefix_tokens:
            model_input.append_int(token_id)

    response = await sampling_client.sample_async(
        model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
        ),
    )

    results = []
    for seq in response.sequences:
        parsed_msg, parse_ok = renderer.parse_response(seq.tokens)
        content = parsed_msg["content"]
        content = extract_content_as_string(content)
        if think_prefix:
            content = "<think>\n" + content
        results.append(content)

    return results


async def test_checkpoint(sampler_path: str, label: str, problems, system_prompt,
                          think_prefix=True, max_tokens=MAX_TOKENS):
    """Test a checkpoint on a few problems."""
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"  Checkpoint: {sampler_path[:80]}...")
    print(f"  think_prefix={think_prefix}, max_tokens={max_tokens}")
    print(f"{'='*70}")

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=sampler_path)

    for i, prob in enumerate(problems):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prob["question"]},
        ]

        try:
            results = await generate_fixed(
                sampling_client, renderer, tokenizer, messages,
                max_tokens=max_tokens, think_prefix=think_prefix,
            )
            response = results[0]
        except Exception as e:
            response = f"ERROR: {e}"

        resp_len = len(response)
        has_boxed = "\\boxed" in response
        has_think = "<think>" in response
        has_end_think = "</think>" in response
        double_think = "<think>\n<think>" in response

        print(f"\n  --- Problem {i+1}: {prob['question'][:60]}... ---")
        print(f"  Expected: {prob['answer']}")
        print(f"  Response length: {resp_len} chars")
        print(f"  Has \\boxed: {has_boxed}, Has <think>: {has_think}, "
              f"Has </think>: {has_end_think}, Double <think>: {double_think}")

        # Show first 500 chars and last 500 chars
        if resp_len <= 1200:
            print(f"  Full response:\n{response}")
        else:
            print(f"  First 500 chars:\n{response[:500]}")
            print(f"  ...[{resp_len - 1000} chars omitted]...")
            print(f"  Last 500 chars:\n{response[-500:]}")


async def test_base_model(problems, system_prompt, think_prefix=True, max_tokens=MAX_TOKENS):
    """Test base model for comparison."""
    print(f"\n{'='*70}")
    print(f"Testing: BASE MODEL")
    print(f"  think_prefix={think_prefix}, max_tokens={max_tokens}")
    print(f"{'='*70}")

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(base_model=MODEL_NAME)

    for i, prob in enumerate(problems):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prob["question"]},
        ]

        try:
            results = await generate_fixed(
                sampling_client, renderer, tokenizer, messages,
                max_tokens=max_tokens, think_prefix=think_prefix,
            )
            response = results[0]
        except Exception as e:
            response = f"ERROR: {e}"

        resp_len = len(response)
        has_boxed = "\\boxed" in response
        has_think = "<think>" in response

        print(f"\n  --- Problem {i+1}: {prob['question'][:60]}... ---")
        print(f"  Expected: {prob['answer']}")
        print(f"  Response length: {resp_len} chars")
        print(f"  Has \\boxed: {has_boxed}")

        if resp_len <= 1200:
            print(f"  Full response:\n{response}")
        else:
            print(f"  First 500 chars:\n{response[:500]}")
            print(f"  ...[{resp_len - 1000} chars omitted]...")
            print(f"  Last 500 chars:\n{response[-500:]}")


def get_checkpoints(task: str, config: str) -> list[dict]:
    """Get all checkpoints for a config."""
    ckpt_file = TRAINING_BASE / task / config / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []
    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return checkpoints


async def main():
    # Load the winning math checkpoint
    winners = json.loads((Path(__file__).parent / "training_winners.json").read_text())
    math_winner = winners["math"]
    code_winner = winners["code"]

    print("MATH WINNER:", json.dumps(math_winner, indent=2))
    print("CODE WINNER:", json.dumps(code_winner, indent=2))

    # ── Test 1: Base model for comparison ──
    await test_base_model(TEST_PROBLEMS, MATH_SYSTEM_PROMPT, think_prefix=True)

    # ── Test 2: Math distill winner with fix (think_prefix=True, default max_tokens) ──
    await test_checkpoint(
        math_winner["sampler_path"],
        "MATH DISTILL D_long (fixed parse, think_prefix=True, max_tokens=8192)",
        TEST_PROBLEMS, MATH_SYSTEM_PROMPT,
        think_prefix=True, max_tokens=8192,
    )

    # ── Test 3: Math distill winner with think_prefix=False ──
    await test_checkpoint(
        math_winner["sampler_path"],
        "MATH DISTILL D_long (fixed parse, think_prefix=FALSE, max_tokens=8192)",
        TEST_PROBLEMS, MATH_SYSTEM_PROMPT,
        think_prefix=False, max_tokens=8192,
    )

    # ── Test 4: Math distill winner with higher max_tokens ──
    await test_checkpoint(
        math_winner["sampler_path"],
        "MATH DISTILL D_long (fixed parse, think_prefix=True, max_tokens=16384)",
        TEST_PROBLEMS, MATH_SYSTEM_PROMPT,
        think_prefix=True, max_tokens=16384,
    )

    # ── Test 5: Try an earlier math checkpoint (step 100 = very early) ──
    math_ckpts = get_checkpoints("math", "D_long")
    early_ckpts = [c for c in math_ckpts if c.get("batch") in [100, 200, 500]]
    for ckpt in early_ckpts:
        step = ckpt["batch"]
        await test_checkpoint(
            ckpt["sampler_path"],
            f"MATH D_long step {step} (fixed parse, think_prefix=True, max_tokens=8192)",
            TEST_PROBLEMS[:2], MATH_SYSTEM_PROMPT,  # Only 2 problems to save time
            think_prefix=True, max_tokens=8192,
        )

    # ── Test 6: Try math A_fast (less training) ──
    a_fast_ckpts = get_checkpoints("math", "A_fast")
    if a_fast_ckpts:
        # Get the best checkpoint for A_fast
        best_a = None
        best_nll = float("inf")
        metrics = []
        mf = TRAINING_BASE / "math" / "A_fast" / "metrics.jsonl"
        with open(mf) as f:
            for line in f:
                d = json.loads(line)
                if "test/nll" in d and d["test/nll"] < best_nll:
                    best_nll = d["test/nll"]
                    best_step = d.get("step", -1)
        for ckpt in a_fast_ckpts:
            if ckpt.get("batch") == best_step and "sampler_path" in ckpt:
                best_a = ckpt
                break
        if best_a is None and a_fast_ckpts:
            best_a = a_fast_ckpts[-1]  # Use final

        if best_a:
            await test_checkpoint(
                best_a["sampler_path"],
                f"MATH A_fast best (step {best_a['batch']}, fixed parse, think_prefix=True)",
                TEST_PROBLEMS[:2], MATH_SYSTEM_PROMPT,
                think_prefix=True, max_tokens=8192,
            )

    print("\n\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())

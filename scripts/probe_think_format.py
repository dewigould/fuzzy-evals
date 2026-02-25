"""
Probe a checkpoint to see if it generates </think> tag.
Usage: python probe_think_format.py <sampler_path_or_config_dir>

Generates 5 responses with <think>\n seeding and checks for </think>.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

PROMPTS = [
    "Solve the following math problem. Put your final answer in \\boxed{}.\n\nWhat is 17 * 23?",
    "Solve the following math problem. Put your final answer in \\boxed{}.\n\nFind the sum of all integers from 1 to 100.",
    "Solve the following math problem. Put your final answer in \\boxed{}.\n\nIf x^2 - 5x + 6 = 0, what are the values of x?",
    "Solve the following math problem. Put your final answer in \\boxed{}.\n\nWhat is the derivative of x^3 + 2x?",
    "Solve the following math problem. Put your final answer in \\boxed{}.\n\nA bag has 3 red and 5 blue balls. What is the probability of drawing a red ball?",
]


async def probe(sampler_path):
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()

    if sampler_path.startswith("tinker://"):
        client = sc.create_sampling_client(model_path=sampler_path)
    else:
        client = sc.create_sampling_client(base_model=MODEL_NAME)

    prefix_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)

    n_has_close = 0
    n_has_boxed = 0

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)
        for tok in prefix_tokens:
            model_input.append_int(tok)

        resp = await client.sample_async(
            model_input, num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=0.0, max_tokens=8192,
                stop=renderer.get_stop_sequences(),
            ),
        )
        parsed, _ = renderer.parse_response(resp.sequences[0].tokens)
        content = "<think>\n" + parsed['content']

        has_close = "</think>" in content
        has_boxed = "\\boxed" in content or "boxed{" in content
        n_has_close += has_close
        n_has_boxed += has_boxed

        print(f"\n--- Prompt {i+1} ---")
        print(f"Has </think>: {has_close} | Has \\boxed: {has_boxed} | Length: {len(content)}")
        # Show first 300 chars
        print(content[:300])
        if len(content) > 300:
            print(f"... [{len(content) - 300} more chars]")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_has_close}/5 have </think>, {n_has_boxed}/5 have \\boxed")
    print(f"{'='*60}")
    return n_has_close, n_has_boxed


def find_latest_checkpoint(config_dir):
    """Find the latest sampler checkpoint in a config directory."""
    ckpt_file = Path(config_dir) / 'checkpoints.jsonl'
    if not ckpt_file.exists():
        return None
    checkpoints = []
    with open(ckpt_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    sampler_ckpts = [c for c in checkpoints if 'sampler_path' in c]
    if not sampler_ckpts:
        return None
    return sampler_ckpts[-1]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python probe_think_format.py <sampler_path_or_config_dir>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.startswith("tinker://"):
        sampler_path = arg
    else:
        ckpt = find_latest_checkpoint(arg)
        if ckpt is None:
            print(f"No checkpoint found in {arg}")
            sys.exit(1)
        sampler_path = ckpt['sampler_path']
        print(f"Using checkpoint: step {ckpt.get('batch', '?')} -> {sampler_path}")

    asyncio.run(probe(sampler_path))

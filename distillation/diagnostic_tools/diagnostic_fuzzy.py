"""Diagnostic: run first weird question on both distilled models with think_prefix=True.

Appends results to diagnostic_outputs.md.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/workspace/tinker-cookbook')

import tinker
from config import FUZZY_SYSTEM_PROMPT, FUZZY_DATA_DIR
from infer import setup_renderer_and_tokenizer, create_checkpoint_client, _content_to_string
from utils import parse_think_tags

MATH_DISTILL_PATH = "tinker://20cab21e-fd37-5d46-92b3-3a9e0855bb90:train:0/sampler_weights/001800"
CODE_DISTILL_PATH = "tinker://f136a2a0-f167-53f9-9e81-db0a13641117:train:0/sampler_weights/001900"


async def run_single(model_name, sampling_client, renderer, tokenizer, question, think_prefix=True, max_tokens=28672):
    messages = [
        {"role": "system", "content": FUZZY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prefill = "<think>\n" if think_prefix else None
    model_input = renderer.build_generation_prompt(messages, prefill=prefill)

    # Decode full prompt
    all_input_tokens = []
    for chunk in model_input.chunks:
        if hasattr(chunk, 'tokens'):
            all_input_tokens.extend(chunk.tokens)
    prompt_text = tokenizer.decode(all_input_tokens)
    n_prompt_tokens = len(all_input_tokens)

    t0 = time.time()
    response = await sampling_client.sample_async(
        model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
        ),
    )
    elapsed = time.time() - t0

    seq = response.sequences[0]
    n_output_tokens = len(seq.tokens)
    raw_decoded = tokenizer.decode(seq.tokens)

    parsed_msg, parse_success = renderer.parse_response(seq.tokens)
    raw_content = parsed_msg["content"]
    content_type = type(raw_content).__name__

    content_parts_debug = None
    if isinstance(raw_content, list):
        content_parts_debug = []
        for i, part in enumerate(raw_content):
            content_parts_debug.append(
                f"  Part {i}: type={type(part).__name__}, "
                f"keys={list(part.keys()) if isinstance(part, dict) else 'N/A'}, "
                f"preview={str(part)[:200]}"
            )

    final_content = _content_to_string(raw_content)
    cot, user_output = parse_think_tags(final_content)

    return {
        "model_name": model_name,
        "task_type": "fuzzy_weird_qs",
        "think_prefix": think_prefix,
        "question": question,
        "system_prompt": FUZZY_SYSTEM_PROMPT,
        "prompt_tail_500": prompt_text[-500:],
        "n_prompt_tokens": n_prompt_tokens,
        "n_output_tokens": n_output_tokens,
        "elapsed_s": elapsed,
        "raw_decoded_first_500": raw_decoded[:500],
        "raw_decoded_last_500": raw_decoded[-500:],
        "parse_success": parse_success,
        "content_type": content_type,
        "content_parts_debug": content_parts_debug,
        "final_content": final_content,
        "cot": cot,
        "user_output": user_output,
    }


def format_result(r):
    lines = []
    lines.append(f"### {r['model_name']} — {r['task_type']} (think_prefix={r['think_prefix']})")
    lines.append("")
    lines.append(f"**Question:** `{r['question']}`")
    lines.append("")
    lines.append(f"**System prompt:** `{r['system_prompt']}`")
    lines.append("")
    lines.append(f"**Prompt tail (last 500 chars of tokenized input):**")
    lines.append(f"```")
    lines.append(r["prompt_tail_500"])
    lines.append(f"```")
    lines.append("")
    lines.append(f"**Generation stats:** {r['n_prompt_tokens']} prompt tokens, {r['n_output_tokens']} output tokens, {r['elapsed_s']:.1f}s")
    lines.append("")
    lines.append(f"**Raw decoded (first 500 chars of seq.tokens):**")
    lines.append(f"```")
    lines.append(r["raw_decoded_first_500"])
    lines.append(f"```")
    lines.append("")
    lines.append(f"**Raw decoded (last 500 chars):**")
    lines.append(f"```")
    lines.append(r["raw_decoded_last_500"])
    lines.append(f"```")
    lines.append("")
    lines.append(f"**parse_response result:** parse_success={r['parse_success']}, content type={r['content_type']}")
    if r["content_parts_debug"]:
        lines.append(f"**ContentPart breakdown:**")
        for p in r["content_parts_debug"]:
            lines.append(p)
    lines.append("")
    lines.append(f"**Final content (_content_to_string output, first 2000 chars):**")
    lines.append(f"```")
    lines.append(r["final_content"][:2000])
    lines.append(f"```")
    lines.append("")
    lines.append(f"**Final content (last 1000 chars):**")
    lines.append(f"```")
    lines.append(r["final_content"][-1000:])
    lines.append(f"```")
    lines.append("")
    lines.append(f"**parse_think_tags result:**")
    lines.append(f"- cot present: {r['cot'] is not None} (length: {len(r['cot']) if r['cot'] else 0})")
    lines.append(f"- user_output (first 1000 chars):")
    lines.append(f"```")
    lines.append((r["user_output"] or "")[:1000])
    lines.append(f"```")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


async def main():
    # Load first weird question
    with open(FUZZY_DATA_DIR / "weird_questions.json") as f:
        questions = json.load(f)
    question = questions[0]["prompt"]
    print(f"Question: {question}")

    renderer, tokenizer = setup_renderer_and_tokenizer()

    print("Creating math_distill client...")
    math_client = create_checkpoint_client(MATH_DISTILL_PATH)
    print("Creating code_distill client...")
    code_client = create_checkpoint_client(CODE_DISTILL_PATH)

    results = []
    for model_name, client in [("math_distill", math_client), ("code_distill", code_client)]:
        print(f"\nRunning: {model_name} / fuzzy_weird_qs / think_prefix=True")
        r = await run_single(model_name, client, renderer, tokenizer, question, think_prefix=True)
        results.append(r)
        print(f"  Done: {r['n_output_tokens']} tokens in {r['elapsed_s']:.1f}s, content_type={r['content_type']}")

    # Append to existing diagnostic_outputs.md
    md_path = Path(__file__).parent / "diagnostic_outputs.md"
    existing = md_path.read_text()

    new_section = []
    new_section.append("\n## Fuzzy Eval Results (weird question #1, think_prefix=True)")
    new_section.append("")
    new_section.append(f"Question: *\"{question}\"*")
    new_section.append("")

    # Summary row
    new_section.append("| Model | Task | think_prefix | content_type | output_tokens |")
    new_section.append("|-------|------|-------------|-------------|---------------|")
    for r in results:
        new_section.append(f"| {r['model_name']} | {r['task_type']} | {r['think_prefix']} | {r['content_type']} | {r['n_output_tokens']} |")
    new_section.append("")

    for r in results:
        new_section.append(format_result(r))

    md_path.write_text(existing + "\n".join(new_section))
    print(f"\nAppended fuzzy results to {md_path}")


if __name__ == "__main__":
    asyncio.run(main())

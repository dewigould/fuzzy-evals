"""Dump the first N training examples exactly as the training pipeline sees them.

Writes a markdown file showing the full system prompt + user + assistant messages
for each example, along with token counts.

Usage:
  python dump_training_examples.py --n 10 --output examples.md
"""

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, "/workspace/tinker-cookbook")

from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT
from experiment_16_02.config import FILTERED_MATH_DATA, FILTERED_CODE_DATA

from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

# Use the base model from experiment config
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"


def build_math_messages(row: dict) -> list[dict]:
    """Build messages exactly as OpenR1MathBuilder._make_map_fn does."""
    messages_raw = row.get("messages", [])
    if not messages_raw:
        messages_raw = [
            {"role": "user", "content": row.get("problem", "")},
            {"role": "assistant", "content": row.get("solution", "")},
        ]
    messages = [{"role": "system", "content": MATH_SYSTEM_PROMPT}]
    for m in messages_raw:
        messages.append({"role": m["role"], "content": m["content"]})
    return messages


def build_code_messages(row: dict) -> list[dict]:
    """Build messages exactly as KodCodeSFTBuilder._make_map_fn does."""
    conversations = row.get("conversations", [])
    user_content = ""
    assistant_content = ""
    for turn in conversations:
        if turn["from"] == "human":
            user_content = turn["value"]
        elif turn["from"] == "gpt":
            assistant_content = turn["value"]
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of examples per dataset")
    parser.add_argument("--output", type=str, default=str(EXPERIMENT_DIR / "results" / "training_examples.md"))
    parser.add_argument("--max-length", type=int, default=4096, help="Training max_length (for tokenization)")
    args = parser.parse_args()

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

    lines = []
    lines.append(f"# Training Examples Dump\n")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Training max_length: {args.max_length}")
    lines.append(f"Examples per dataset: {args.n}\n")

    # ── Math examples ──
    lines.append("---\n")
    lines.append("# Math Distillation Examples\n")
    lines.append(f"System prompt: `{MATH_SYSTEM_PROMPT}`\n")
    lines.append(f"Data source: `{FILTERED_MATH_DATA}`\n")

    with open(FILTERED_MATH_DATA) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            row = json.loads(line)
            messages = build_math_messages(row)

            # Tokenize at training max_length (what actually gets trained)
            datum_train = conversation_to_datum(messages, renderer, args.max_length, train_on_what)
            train_tokens = datum_train.model_input.length

            # Tokenize at large max_length to get true length
            datum_full = conversation_to_datum(messages, renderer, 32768, train_on_what)
            full_tokens = datum_full.model_input.length

            truncated = full_tokens > args.max_length

            lines.append(f"## Math Example {i+1}\n")
            lines.append(f"**Token count**: {full_tokens} tokens (full) | {train_tokens} tokens (at max_length={args.max_length})")
            if truncated:
                lines.append(f"  **TRUNCATED** (loses {full_tokens - train_tokens} tokens)")
            lines.append("")

            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                lines.append(f"### [{role}]\n")
                # Check if content contains think tags
                if "<think>" in content:
                    # Split into think and non-think parts for readability
                    lines.append(content)
                else:
                    lines.append(content)
                lines.append("")

            lines.append("---\n")

    # ── Code examples ──
    lines.append("\n# Code Distillation Examples\n")
    lines.append(f"System prompt: `{CODE_SYSTEM_PROMPT}`\n")
    lines.append(f"Data source: `{FILTERED_CODE_DATA}`\n")

    with open(FILTERED_CODE_DATA) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            row = json.loads(line)
            messages = build_code_messages(row)

            datum_train = conversation_to_datum(messages, renderer, args.max_length, train_on_what)
            train_tokens = datum_train.model_input.length

            datum_full = conversation_to_datum(messages, renderer, 32768, train_on_what)
            full_tokens = datum_full.model_input.length

            truncated = full_tokens > args.max_length

            lines.append(f"## Code Example {i+1}\n")
            lines.append(f"**Token count**: {full_tokens} tokens (full) | {train_tokens} tokens (at max_length={args.max_length})")
            if truncated:
                lines.append(f"  **TRUNCATED** (loses {full_tokens - train_tokens} tokens)")
            lines.append("")

            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                lines.append(f"### [{role}]\n")
                lines.append(content)
                lines.append("")

            lines.append("---\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {output_path}")

    # Summary stats
    print(f"\nDumped {args.n} math + {args.n} code examples")


if __name__ == "__main__":
    main()

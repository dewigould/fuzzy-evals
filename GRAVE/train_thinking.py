"""SFT training on thinking-model-generated reasoning traces.

Trains on correct_only.jsonl from the thinking model trace generation pipeline.
Supports both math and code tasks, and both base and instruct models.

Usage:
  python train_thinking.py --task math --model-type base
  python train_thinking.py --task math --model-type it
  python train_thinking.py --task code --model-type base
  python train_thinking.py --task code --model-type it
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import cast

import chz
import datasets
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    StreamingSupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

from config import TRAINING_BASE, LORA_RANK, MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Base"
IT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

TRAIN_MAX_LENGTH = 4096
TEST_SIZE = 200
BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42

# Default config: A_r32_high_lr
DEFAULT_LR = 2e-4
DEFAULT_STEPS = 500
DEFAULT_BATCH_SIZE = 50
DEFAULT_SCHEDULE = "cosine"

# Paths to thinking trace data
THINKING_MATH_DATA = Path(__file__).parent / "generate_reasoning_traces" / "data_thinking" / "correct_only.jsonl"
THINKING_CODE_DATA = Path(__file__).parent / "generate_reasoning_traces" / "data_code_thinking" / "correct_only.jsonl"


class FixedListDataset(SupervisedDataset):
    """In-memory dataset for test split (NLL evaluation)."""

    def __init__(self, data: list[tinker.Datum], batch_size: int):
        self._data = data
        self._batch_size = batch_size

    def __len__(self):
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def get_batch(self, idx):
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._data))
        return self._data[start:end]


@chz.chz
class ThinkingTraceBuilder(ChatDatasetBuilder):
    """SFT dataset builder for thinking-model-generated traces."""
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 25_000
    test_size: int = TEST_SIZE
    data_path: str = ""
    system_prompt: str = ""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        data_path = Path(self.data_path)
        print(f"Loading thinking traces from {data_path}")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Thinking trace data not found at {data_path}. "
                "Run the trace generation + grading pipeline first."
            )

        n_lines = sum(1 for _ in open(data_path))
        print(f"  {n_lines:,} correct traces available")

        map_fn = self._make_map_fn()

        ds = datasets.load_dataset(
            "json", data_files=str(data_path), split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        print(f"  Collecting {self.test_size} test examples...")
        test_rows = []
        remaining_rows = []
        row_count = 0
        for row in ds:
            if row_count < self.test_size:
                test_rows.append(row)
            else:
                remaining_rows.append(row)
                if len(remaining_rows) >= self.buffer_size:
                    break
            row_count += 1

        test_datums = []
        skipped = 0
        for row in test_rows:
            try:
                datum = map_fn(row)
                test_datums.append(datum)
            except Exception:
                skipped += 1
        print(f"  Test set: {len(test_datums)} datums ({skipped} skipped)")

        def train_row_generator():
            for row in remaining_rows:
                yield row
            ds2 = datasets.load_dataset(
                "json", data_files=str(data_path), split="train", streaming=True
            )
            ds2 = cast(datasets.IterableDataset, ds2)
            ds2 = ds2.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)
            skip_count = 0
            for row in ds2:
                skip_count += 1
                if skip_count <= self.test_size + len(remaining_rows):
                    continue
                yield row

        train_iter = datasets.IterableDataset.from_generator(train_row_generator)
        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=train_iter,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,
            map_fn=map_fn,
            buffer_size=self.buffer_size,
        )

        test_dataset = FixedListDataset(test_datums, batch_size=self.test_size)
        return train_dataset, test_dataset

    def _make_map_fn(self):
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
        system_prompt = self.system_prompt

        def map_fn(row: dict) -> tinker.Datum:
            # The correct_only.jsonl has a 'messages' field: [user, assistant]
            messages_raw = row.get("messages", [])
            if not messages_raw:
                # Fallback: reconstruct from problem/generation or question/generation
                if "problem" in row:
                    messages_raw = [
                        {"role": "user", "content": row["problem"]},
                        {"role": "assistant", "content": row["generation"]},
                    ]
                elif "question" in row:
                    messages_raw = [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["generation"]},
                    ]

            messages = [{"role": "system", "content": system_prompt}]
            for m in messages_raw:
                messages.append({"role": m["role"], "content": m["content"]})

            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )
        return map_fn


def main():
    parser = argparse.ArgumentParser(
        description="Train on thinking-model-generated reasoning traces"
    )
    parser.add_argument(
        "--task", required=True, choices=["math", "code"],
        help="Task: math or code"
    )
    parser.add_argument(
        "--model-type", required=True, choices=["base", "it"],
        help="Model type: base (Qwen3-30B-A3B-Base) or it (Qwen3-30B-A3B-Instruct-2507)"
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    args = parser.parse_args()

    # Select model
    if args.model_type == "base":
        model_name = BASE_MODEL
        model_tag = "base"
    else:
        model_name = IT_MODEL
        model_tag = "it"

    # Select task data
    if args.task == "math":
        data_path = str(THINKING_MATH_DATA)
        system_prompt = MATH_SYSTEM_PROMPT
    else:
        data_path = str(THINKING_CODE_DATA)
        system_prompt = CODE_SYSTEM_PROMPT

    max_prompts = args.steps * args.batch_size
    num_epochs = 1
    # If more prompts needed than available data, increase epochs
    try:
        n_available = sum(1 for _ in open(data_path))
        if max_prompts > n_available:
            num_epochs = (max_prompts // n_available) + 1
            print(f"  Need {num_epochs} epochs for {max_prompts} prompts ({n_available} available)")
    except FileNotFoundError:
        pass  # Will be caught later by the builder

    task_name = f"thinking_{args.task}_{model_tag}"
    log_path = f"{TRAINING_BASE}/{task_name}/A_r32_high_lr"

    print(f"Thinking Trace Training: {args.task} on {model_tag}")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_path}")
    print(f"  LR: {args.lr}, Steps: {args.steps}, Schedule: {args.schedule}")
    print(f"  Batch size: {args.batch_size}, Train max_length: {TRAIN_MAX_LENGTH}")
    print(f"  Max prompts: {max_prompts}")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=TRAIN_MAX_LENGTH,
        batch_size=args.batch_size,
    )

    dataset_builder = ThinkingTraceBuilder(
        common_config=common_config,
        max_prompts=max_prompts,
        test_size=TEST_SIZE,
        data_path=data_path,
        system_prompt=system_prompt,
    )

    config = train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule=args.schedule,
        num_epochs=num_epochs,
        lora_rank=LORA_RANK,
        save_every=40,
        eval_every=40,
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete: {task_name}")


if __name__ == "__main__":
    main()

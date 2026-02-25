"""Unified training script for the 25_02 sweep.

Single entry point for any (base_model, task, traces) training run.
Loads pre-filtered data from filtered_data/, trains with LoRA via tinker.

Usage:
  python train.py --base-model llama8b --task math --traces sonnet
  python train.py --base-model qwen_it --task code --traces kimi --max-length 8192
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep_25_02.config import (
    BASE_MODELS, TRACE_SOURCES, SYSTEM_PROMPTS,
    DEFAULT_MAX_LENGTH, DEFAULT_LORA_RANK, DEFAULT_LR,
    DEFAULT_BATCH_SIZE, DEFAULT_SCHEDULE, DEFAULT_MAX_PROMPTS,
    DEFAULT_SAVE_EVERY, DEFAULT_EVAL_EVERY,
    TEST_SIZE, SHUFFLE_SEED, BUFFER_SIZE,
    TRAINING_BASE,
    build_run_name, build_filtered_data_path,
)


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
class FilteredTraceBuilder(ChatDatasetBuilder):
    """SFT dataset builder reading from pre-filtered JSONL."""
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = DEFAULT_MAX_PROMPTS
    test_size: int = TEST_SIZE
    data_path: str = ""
    system_prompt: str = ""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        data_path = Path(self.data_path)
        print(f"Loading filtered traces from {data_path}")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Filtered data not found at {data_path}. "
                "Run filter_data.py first."
            )

        n_lines = sum(1 for _ in open(data_path))
        print(f"  {n_lines:,} filtered examples available")

        map_fn = self._make_map_fn()

        ds = datasets.load_dataset(
            "json", data_files=str(data_path), split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        # Collect test examples
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
            messages_raw = row.get("messages", [])
            if not messages_raw:
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
    parser = argparse.ArgumentParser(description="Train a single sweep run")
    parser.add_argument("--base-model", required=True, choices=list(BASE_MODELS.keys()),
                        help="Base model short name")
    parser.add_argument("--task", required=True, choices=["math", "code"],
                        help="Task: math or code")
    parser.add_argument("--traces", required=True, choices=["sonnet", "qwen", "kimi"],
                        help="Trace source")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Override max prompts (default: min(available, 25000))")
    args = parser.parse_args()

    base_model = BASE_MODELS[args.base_model]
    system_prompt = SYSTEM_PROMPTS[args.task]
    run_name = build_run_name(
        args.base_model, args.task, args.traces,
        args.max_length, args.lora_rank, args.lr,
    )
    filtered_path = build_filtered_data_path(
        args.base_model, args.task, args.traces, args.max_length,
    )

    if not filtered_path.exists():
        print(f"ERROR: Filtered data not found at {filtered_path}")
        print("Run filter_data.py first.")
        sys.exit(1)

    # Count available examples
    n_available = sum(1 for _ in open(filtered_path))
    max_prompts = args.max_prompts or min(n_available, DEFAULT_MAX_PROMPTS)
    steps = max_prompts // args.batch_size
    num_epochs = 1

    log_path = str(TRAINING_BASE / run_name)

    print(f"Sweep 25_02 Training: {run_name}")
    print(f"  Base model: {base_model}")
    print(f"  Task: {args.task}, Traces: {args.traces}")
    print(f"  Filtered data: {filtered_path} ({n_available:,} examples)")
    print(f"  Max prompts: {max_prompts:,}, Steps: {steps}")
    print(f"  LR: {args.lr}, Schedule: {args.schedule}")
    print(f"  Batch size: {args.batch_size}, Max length: {args.max_length}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(base_model)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    dataset_builder = FilteredTraceBuilder(
        common_config=common_config,
        max_prompts=max_prompts,
        test_size=TEST_SIZE,
        data_path=str(filtered_path),
        system_prompt=system_prompt,
    )

    config = train.Config(
        log_path=log_path,
        model_name=base_model,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule=args.schedule,
        num_epochs=num_epochs,
        lora_rank=args.lora_rank,
        save_every=DEFAULT_SAVE_EVERY,
        eval_every=DEFAULT_EVAL_EVERY,
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete: {run_name}")


if __name__ == "__main__":
    main()

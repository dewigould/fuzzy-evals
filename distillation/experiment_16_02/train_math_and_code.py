"""Experiment 16/02 — Interleaved math+code distillation training.

SFT on interleaved Sonnet math and code traces into a base model.
Alternates between math and code examples within each batch.

Usage:
  # Default: 25K samples (12.5K math + 12.5K code), lr=2e-4
  python train_math_and_code.py --name sonnet_mixed_qwen_4k

  # Quick smoke test
  python train_math_and_code.py --name smoke --samples 500 --save-every 1 --batch-size 10
"""

import argparse
import asyncio
import itertools
import random
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

# Add parent distillation/ to path
EXPERIMENT_DIR = Path(__file__).parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))

from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT  # noqa: E402
from experiment_16_02.config import (  # noqa: E402
    MODEL_NAME, LORA_RANK, TRAIN_MAX_LENGTH, TRAIN_LR,
    TRAIN_SAMPLES, TRAIN_BATCH_SIZE, TRAIN_SCHEDULE,
    SAVE_EVERY, EVAL_EVERY, TRAINING_DIR,
)

# Sonnet trace data paths
SONNET_MATH_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data" / "correct_only.jsonl"
SONNET_CODE_DATA = DISTILLATION_DIR / "generate_reasoning_traces" / "data_code" / "correct_only.jsonl"

TEST_SIZE = 200
BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42


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
class InterleavedMathCodeBuilder(ChatDatasetBuilder):
    """SFT dataset builder that interleaves Sonnet math and code traces.

    Alternates: math, code, math, code, ... to produce a mixed-domain
    training set. Each example uses its domain-appropriate system prompt.
    """
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 25_000
    test_size: int = TEST_SIZE
    filter_max_length: int | None = None
    math_data_path: str | None = None
    code_data_path: str | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        math_path = Path(self.math_data_path) if self.math_data_path else SONNET_MATH_DATA
        code_path = Path(self.code_data_path) if self.code_data_path else SONNET_CODE_DATA

        print(f"Loading interleaved math+code data")
        print(f"  Math: {math_path}")
        print(f"  Code: {code_path}")

        for p in [math_path, code_path]:
            if not p.exists():
                raise FileNotFoundError(f"Data not found at {p}")

        math_n = sum(1 for _ in open(math_path))
        code_n = sum(1 for _ in open(code_path))
        print(f"  Math: {math_n:,} examples, Code: {code_n:,} examples")

        map_fn = self._make_map_fn()
        length_filter = self._make_length_filter() if self.filter_max_length else None
        if length_filter:
            print(f"  Filtering to examples ≤ {self.filter_max_length} tokens")

        # Load both datasets with shuffling
        math_ds = datasets.load_dataset(
            "json", data_files=str(math_path), split="train", streaming=True
        )
        math_ds = cast(datasets.IterableDataset, math_ds)
        math_ds = math_ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        code_ds = datasets.load_dataset(
            "json", data_files=str(code_path), split="train", streaming=True
        )
        code_ds = cast(datasets.IterableDataset, code_ds)
        code_ds = code_ds.shuffle(seed=SHUFFLE_SEED + 1, buffer_size=self.buffer_size)

        # Interleave: tag each row with its domain
        def tagged_iter(ds_iter, domain: str):
            for row in ds_iter:
                row["_domain"] = domain
                yield row

        math_iter = tagged_iter(iter(math_ds), "math")
        code_iter = tagged_iter(iter(code_ds), "code")

        # Collect test set (interleaved)
        print(f"  Collecting {self.test_size} test examples (interleaved)...")
        test_rows = []
        remaining_rows = []
        row_count = 0
        filtered_count = 0

        for math_row, code_row in zip(math_iter, code_iter):
            for row in [math_row, code_row]:
                if length_filter and not length_filter(row):
                    filtered_count += 1
                    continue
                if row_count < self.test_size:
                    test_rows.append(row)
                else:
                    remaining_rows.append(row)
                    if len(remaining_rows) >= self.buffer_size:
                        break
                row_count += 1
            if len(remaining_rows) >= self.buffer_size:
                break

        if filtered_count:
            print(f"  Filtered out {filtered_count} examples exceeding {self.filter_max_length} tokens")

        test_datums = []
        skipped = 0
        for row in test_rows:
            try:
                test_datums.append(map_fn(row))
            except Exception:
                skipped += 1
        math_test = sum(1 for r in test_rows if r.get("_domain") == "math")
        code_test = sum(1 for r in test_rows if r.get("_domain") == "code")
        print(f"  Test set: {len(test_datums)} datums ({math_test} math, {code_test} code, {skipped} skipped)")

        # Build training iterator
        def train_row_generator():
            # First yield buffered rows
            for row in remaining_rows:
                yield row

            # Then re-stream both datasets interleaved
            math_ds2 = datasets.load_dataset(
                "json", data_files=str(math_path), split="train", streaming=True
            )
            math_ds2 = cast(datasets.IterableDataset, math_ds2)
            math_ds2 = math_ds2.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

            code_ds2 = datasets.load_dataset(
                "json", data_files=str(code_path), split="train", streaming=True
            )
            code_ds2 = cast(datasets.IterableDataset, code_ds2)
            code_ds2 = code_ds2.shuffle(seed=SHUFFLE_SEED + 1, buffer_size=self.buffer_size)

            math_iter2 = tagged_iter(iter(math_ds2), "math")
            code_iter2 = tagged_iter(iter(code_ds2), "code")

            skip_count = 0
            for math_row, code_row in zip(math_iter2, code_iter2):
                for row in [math_row, code_row]:
                    if length_filter and not length_filter(row):
                        continue
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

    def _make_length_filter(self):
        """Create a filter that checks if tokenized example fits within filter_max_length."""
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
        limit = self.filter_max_length

        def fits(row: dict) -> bool:
            domain = row.get("_domain", "math")
            sys_prompt = MATH_SYSTEM_PROMPT if domain == "math" else CODE_SYSTEM_PROMPT
            messages_raw = row.get("messages", [])
            if not messages_raw:
                # Fallback: construct from raw fields
                problem = row.get("problem", row.get("question", ""))
                solution = row.get("solution", row.get("generation", ""))
                messages_raw = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            messages = [{"role": "system", "content": sys_prompt}]
            for m in messages_raw:
                messages.append({"role": m["role"], "content": m["content"]})
            datum = conversation_to_datum(
                messages, self.renderer, 32768, train_on_what
            )
            return datum.model_input.length <= limit
        return fits

    def _make_map_fn(self):
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        def map_fn(row: dict) -> tinker.Datum:
            domain = row.get("_domain", "math")
            sys_prompt = MATH_SYSTEM_PROMPT if domain == "math" else CODE_SYSTEM_PROMPT

            messages_raw = row.get("messages", [])
            if not messages_raw:
                problem = row.get("problem", row.get("question", ""))
                solution = row.get("solution", row.get("generation", ""))
                messages_raw = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            messages = [{"role": "system", "content": sys_prompt}]
            for m in messages_raw:
                messages.append({"role": m["role"], "content": m["content"]})
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )
        return map_fn


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 16/02: Interleaved math+code distillation training"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Run name (used for training_runs/{name}/ and results)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help=f"Base model (default: {MODEL_NAME})")
    parser.add_argument("--lr", type=float, default=TRAIN_LR,
                        help=f"Learning rate (default: {TRAIN_LR})")
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES,
                        help=f"Total training samples (default: {TRAIN_SAMPLES})")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE,
                        help=f"Batch size (default: {TRAIN_BATCH_SIZE})")
    parser.add_argument("--max-length", type=int, default=TRAIN_MAX_LENGTH,
                        help=f"Training max token length (default: {TRAIN_MAX_LENGTH})")
    parser.add_argument("--schedule", type=str, default=TRAIN_SCHEDULE,
                        choices=["cosine", "linear", "constant"],
                        help=f"LR schedule (default: {TRAIN_SCHEDULE})")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help=f"Save checkpoint every N steps (default: {SAVE_EVERY})")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                        help=f"Eval NLL every N steps (default: {EVAL_EVERY})")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--filter-max-length", type=int, default=None,
                        help="Skip training examples exceeding this token count (default: no filtering)")
    args = parser.parse_args()

    steps = args.samples // args.batch_size
    log_path = str(TRAINING_DIR / args.name)

    print(f"Experiment 16/02 — Interleaved Math+Code Distillation Training")
    print(f"  Run name: {args.name}")
    print(f"  Model: {args.model_name}")
    print(f"  Samples: {args.samples} ({steps} steps, ~{args.samples//2} math + ~{args.samples//2} code)")
    print(f"  LR: {args.lr}, Schedule: {args.schedule}")
    print(f"  Batch size: {args.batch_size}, Max length: {args.max_length}")
    print(f"  Save every: {args.save_every} steps ({args.save_every * args.batch_size} samples)")
    print(f"  LoRA rank: {args.lora_rank}")
    if args.filter_max_length:
        print(f"  Filter max length: {args.filter_max_length} tokens (skip longer examples)")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(args.model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    dataset_builder = InterleavedMathCodeBuilder(
        common_config=common_config,
        max_prompts=args.samples,
        test_size=TEST_SIZE,
        filter_max_length=args.filter_max_length,
    )

    config = train.Config(
        log_path=log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule=args.schedule,
        num_epochs=1,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete: {args.name}")


if __name__ == "__main__":
    main()

"""Short code SFT on best math checkpoint to teach code output format.

Loads the best math checkpoint (4K filtered step 200) and does a few steps
of SFT on filtered KodCode data to teach the model to output fenced code
blocks after </think>.

Usage:
  python train_math_code_finetune.py --n-examples 500
  python train_math_code_finetune.py --n-examples 100

Saves to training_runs/math_code_finetune_{n}ex/
"""

import argparse
import asyncio
import json
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

from config import MAX_TOKENS, TRAINING_BASE, LORA_RANK, CODE_SYSTEM_PROMPT

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"

# Best math checkpoint (4K filtered, step 200)
MATH_CHECKPOINT_STATE = "tinker://7303365f-60eb-5fd9-99ef-5b43304ec1e4:train:0/weights/000200"

FILTERED_DATA_PATH = Path(__file__).parent / "filtered_data_kodcode" / "clean.jsonl"
BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42
TEST_SIZE = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_LR = 5e-5  # Low LR to avoid catastrophic forgetting


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
class CodeFineTuneBuilder(ChatDatasetBuilder):
    """Small code SFT dataset for format learning."""
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 500
    test_size: int = TEST_SIZE

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        print(f"Loading filtered KodCode data from {FILTERED_DATA_PATH}")
        if not FILTERED_DATA_PATH.exists():
            raise FileNotFoundError(f"Filtered data not found at {FILTERED_DATA_PATH}")

        map_fn = self._make_map_fn()

        ds = datasets.load_dataset(
            "json", data_files=str(FILTERED_DATA_PATH), split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        # Collect test examples first
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

        def map_fn(row: dict) -> tinker.Datum:
            conversations = row.get("conversations", [])
            user_content = ""
            assistant_content = ""
            for turn in conversations:
                if turn["from"] == "human":
                    user_content = turn["value"]
                elif turn["from"] == "gpt":
                    assistant_content = turn["value"]

            messages = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )
        return map_fn


def main():
    parser = argparse.ArgumentParser(description="Code format fine-tune on math checkpoint")
    parser.add_argument("--n-examples", type=int, required=True,
                        help="Number of code examples to train on (e.g. 10, 100, 500)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: min(n_examples, 50))")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for task name (e.g. 'low_lr')")
    args = parser.parse_args()

    n_examples = args.n_examples
    batch_size = args.batch_size or min(n_examples, DEFAULT_BATCH_SIZE)
    lr = args.lr or DEFAULT_LR
    steps = n_examples // batch_size
    if steps < 1:
        steps = 1

    suffix = f"_{args.suffix}" if args.suffix else ""
    task_name = f"math_code_finetune_{n_examples}ex{suffix}"
    log_path = f"{TRAINING_BASE}/{task_name}"

    print(f"Code Format Fine-Tune on Math Checkpoint")
    print(f"  Base checkpoint: {MATH_CHECKPOINT_STATE}")
    print(f"  Examples: {n_examples}, Steps: {steps}, Batch: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_TOKENS,
        batch_size=batch_size,
    )

    dataset_builder = CodeFineTuneBuilder(
        common_config=common_config,
        max_prompts=n_examples,
        test_size=TEST_SIZE,
    )

    config = train.Config(
        log_path=log_path,
        model_name=BASE_MODEL_NAME,
        load_checkpoint_path=MATH_CHECKPOINT_STATE,
        dataset_builder=dataset_builder,
        learning_rate=lr,
        lr_schedule="cosine",
        num_epochs=1,
        lora_rank=LORA_RANK,
        save_every=1,    # Save every step so we can eval each
        eval_every=1,    # Eval NLL every step
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete: {task_name}")


if __name__ == "__main__":
    main()

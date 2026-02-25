"""Code distillation: SFT on KodCode/KodCode-V1-SFT-R1.

Trains Qwen3-30B-A3B-Instruct-2507 on <think>...</think> code reasoning
traces from KodCode's R1-style dataset (~268K examples).

Usage:
  python train_kodcode.py <config_name>

Config names: A_fast, B_medium, C_gentle, D_long
"""

import asyncio
import sys
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

from config import MODEL_NAME, MAX_TOKENS, TRAINING_BASE, LORA_RANK, CODE_SYSTEM_PROMPT
from sweep_configs import CODE_CONFIGS

DATASET_NAME = "KodCode/KodCode-V1-SFT-R1"
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
class KodCodeSFTBuilder(ChatDatasetBuilder):
    """SFT dataset builder for KodCode-V1-SFT-R1 with streaming and test holdout."""
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 25_000
    test_size: int = TEST_SIZE

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        print(f"Loading {DATASET_NAME} (streaming, shuffled, max_prompts={self.max_prompts})...")
        ds = datasets.load_dataset(DATASET_NAME, split="train", streaming=True)
        ds = cast(datasets.IterableDataset, ds)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        def map_fn(row: dict) -> tinker.Datum:
            conversations = row.get("conversations", [])
            # Extract human/gpt turns from conversations
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

        # Convert test rows to datums (skip if too long)
        test_datums = []
        skipped = 0
        for row in test_rows:
            try:
                datum = map_fn(row)
                test_datums.append(datum)
            except Exception as e:
                skipped += 1
        print(f"  Test set: {len(test_datums)} datums ({skipped} skipped)")

        # Build train dataset from remaining stream
        def train_row_generator():
            for row in remaining_rows:
                yield row
            # Continue streaming (need to re-stream and skip already-seen)
            ds2 = datasets.load_dataset(DATASET_NAME, split="train", streaming=True)
            ds2 = cast(datasets.IterableDataset, ds2)
            ds2 = ds2.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)
            skip_target = self.test_size + len(remaining_rows)
            skip_count = 0
            for row in ds2:
                skip_count += 1
                if skip_count <= skip_target:
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config_name>")
        print(f"Available configs: {', '.join(CODE_CONFIGS.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    if config_name not in CODE_CONFIGS:
        print(f"Unknown config: {config_name}")
        print(f"Available configs: {', '.join(CODE_CONFIGS.keys())}")
        sys.exit(1)

    cfg = CODE_CONFIGS[config_name]
    max_prompts = cfg["steps"] * cfg["batch_size"]
    num_epochs = 1

    log_path = f"{TRAINING_BASE}/kodcode/{config_name}"

    print(f"KodCode Distillation Training: {config_name}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LR: {cfg['lr']}, Steps: {cfg['steps']}, Schedule: {cfg['schedule']}")
    print(f"  Batch size: {cfg['batch_size']}, Max tokens: {MAX_TOKENS}")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_TOKENS,
        batch_size=cfg["batch_size"],
    )

    dataset_builder = KodCodeSFTBuilder(
        common_config=common_config,
        max_prompts=max_prompts,
        test_size=TEST_SIZE,
    )

    config = train.Config(
        log_path=log_path,
        model_name=MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=cfg["lr"],
        lr_schedule=cfg["schedule"],
        num_epochs=num_epochs,
        lora_rank=LORA_RANK,
        save_every=100,
        eval_every=100,
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete for {config_name}")


if __name__ == "__main__":
    main()

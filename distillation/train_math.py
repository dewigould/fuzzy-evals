"""Math distillation: SFT on OpenR1-Math-220k reasoning traces.

Trains Qwen3-30B-A3B-Instruct-2507 on R1-style <think>...</think> math
reasoning traces with a math system prompt.

Usage:
  python train_math.py <config_name>

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

from config import MODEL_NAME, MAX_TOKENS, TRAINING_BASE, LORA_RANK, MATH_SYSTEM_PROMPT
from sweep_configs import MATH_CONFIGS

TEST_SIZE = 200
BUFFER_SIZE = 10_000


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
class OpenR1MathBuilder(ChatDatasetBuilder):
    """SFT dataset builder for OpenR1-Math-220k with streaming and test holdout."""
    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 25_000
    test_size: int = TEST_SIZE

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        print(f"Loading OpenR1-Math-220k (streaming, max_prompts={self.max_prompts})...")
        ds = datasets.load_dataset(
            "open-r1/OpenR1-Math-220k", split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)

        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        def map_fn(row: dict) -> tinker.Datum:
            messages_raw = row.get("messages", [])
            if not messages_raw:
                messages_raw = [
                    {"role": "user", "content": row.get("problem", "")},
                    {"role": "assistant", "content": row.get("solution", "")},
                ]
            # Inject system prompt
            messages = [{"role": "system", "content": MATH_SYSTEM_PROMPT}]
            for m in messages_raw:
                messages.append({"role": m["role"], "content": m["content"]})
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
            ds2 = datasets.load_dataset(
                "open-r1/OpenR1-Math-220k", split="train", streaming=True
            )
            ds2 = cast(datasets.IterableDataset, ds2)
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config_name>")
        print(f"Available configs: {', '.join(MATH_CONFIGS.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    if config_name not in MATH_CONFIGS:
        print(f"Unknown config: {config_name}")
        print(f"Available configs: {', '.join(MATH_CONFIGS.keys())}")
        sys.exit(1)

    cfg = MATH_CONFIGS[config_name]
    max_prompts = cfg["steps"] * cfg["batch_size"]
    num_epochs = 1
    if max_prompts > 90_000:
        num_epochs = (max_prompts // 90_000) + 1
        print(f"  Need {num_epochs} epochs for {max_prompts} prompts (~93K dataset)")

    log_path = f"{TRAINING_BASE}/math/{config_name}"

    print(f"Math Distillation Training: {config_name}")
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

    dataset_builder = OpenR1MathBuilder(
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

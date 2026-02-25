"""
Math SFT single config training on OpenR1-Math-220k reasoning traces.

Usage:
  python run_math_sft_single.py <config_name>

Config names: A_short, B_medium, C_gentle, D_long

Trains model on R1-style <think>...</think> reasoning traces with max_length=8192
to capture complete think-answer pairs.
"""

import asyncio
import sys
from typing import cast

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

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
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = '/workspace/results_10_02/math_sft'
MAX_LENGTH = 8192
TEST_SIZE = 200

# Sweep configurations
CONFIGS = {
    'A_short': {
        'lr': 1e-4,
        'steps': 500,
        'schedule': 'linear',
        'batch_size': 50,
    },
    'B_medium': {
        'lr': 1e-4,
        'steps': 1000,
        'schedule': 'linear',
        'batch_size': 50,
    },
    'C_gentle': {
        'lr': 5e-5,
        'steps': 1000,
        'schedule': 'cosine',
        'batch_size': 50,
    },
    'D_long': {
        'lr': 5e-5,
        'steps': 2000,
        'schedule': 'cosine',
        'batch_size': 50,
    },
}


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
    buffer_size: int = 10_000
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
            messages = row.get("messages", [])
            if not messages:
                messages = [
                    {"role": "user", "content": row.get("problem", "")},
                    {"role": "assistant", "content": row.get("solution", "")},
                ]
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        # Collect test examples first from the stream
        print(f"  Collecting {self.test_size} test examples...")
        test_rows = []
        remaining_stream_rows = []
        row_count = 0
        for row in ds:
            if row_count < self.test_size:
                test_rows.append(row)
            else:
                remaining_stream_rows.append(row)
                if len(remaining_stream_rows) >= self.buffer_size:
                    break
            row_count += 1

        # Convert test rows to datums
        test_datums = []
        for row in test_rows:
            try:
                datum = map_fn(row)
                test_datums.append(datum)
            except Exception as e:
                print(f"  Skipping test example: {e}")
        print(f"  Test set: {len(test_datums)} datums")

        # Build train dataset from remaining stream
        # Create a new iterable that starts with buffered rows then continues streaming
        def train_row_generator():
            for row in remaining_stream_rows:
                yield row
            # Continue from where we left off in the original stream
            # Note: since the original ds is an iterator, it continues from where we stopped
            # But since we already consumed it above, we need to reload
            ds2 = datasets.load_dataset(
                "open-r1/OpenR1-Math-220k", split="train", streaming=True
            )
            ds2 = cast(datasets.IterableDataset, ds2)
            skipped = 0
            for row in ds2:
                skipped += 1
                if skipped <= self.test_size + len(remaining_stream_rows):
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
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    if config_name not in CONFIGS:
        print(f"Unknown config: {config_name}")
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[config_name]
    max_prompts = cfg['steps'] * cfg['batch_size']
    num_epochs = 1
    # D_long: 2000 steps * 50 = 100K prompts, but dataset has ~93K → need >1 epoch
    if max_prompts > 90_000:
        num_epochs = (max_prompts // 90_000) + 1
        print(f"  Need {num_epochs} epochs to cover {max_prompts} prompts (dataset ~93K)")

    log_path = f'{RESULTS_DIR}/{config_name}'

    print(f"Math SFT Training: {config_name}")
    print(f"  LR: {cfg['lr']}")
    print(f"  Steps: {cfg['steps']}")
    print(f"  Schedule: {cfg['schedule']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Max prompts: {max_prompts}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Num epochs: {num_epochs}")
    print(f"  Log path: {log_path}")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=cfg['batch_size'],
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
        learning_rate=cfg['lr'],
        lr_schedule=cfg['schedule'],
        num_epochs=num_epochs,
        lora_rank=32,
        save_every=100,
        eval_every=100,
    )

    asyncio.run(train.main(config))
    print(f"\nTraining complete for {config_name}")


if __name__ == '__main__':
    main()

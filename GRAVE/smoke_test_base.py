"""Smoke test: verify base model distillation pipeline works.

Loads 10 examples each for math and code, prints training data, runs 1 step,
and prints the loss.

Usage:
  python smoke_test_base.py
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

from config import MAX_TOKENS, TRAINING_BASE, LORA_RANK, MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"
SMOKE_EXAMPLES = 10


class FixedListDataset(SupervisedDataset):
    def __init__(self, data: list[tinker.Datum], batch_size: int):
        self._data = data
        self._batch_size = batch_size

    def __len__(self):
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def get_batch(self, idx):
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._data))
        return self._data[start:end]


def build_math_data(renderer, common_config):
    """Load 10 math examples, print them, return as dataset."""
    print(f"\n{'='*60}")
    print("MATH DATA (OpenR1-Math-220k)")
    print(f"{'='*60}")

    ds = datasets.load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    ds = cast(datasets.IterableDataset, ds)

    train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    datums = []
    for i, row in enumerate(ds):
        if i >= SMOKE_EXAMPLES:
            break

        messages_raw = row.get("messages", [])
        if not messages_raw:
            messages_raw = [
                {"role": "user", "content": row.get("problem", "")},
                {"role": "assistant", "content": row.get("solution", "")},
            ]

        # Print the example
        print(f"\n--- Math Example {i+1} ---")
        for m in messages_raw:
            role = m["role"]
            content = m["content"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{role}]: {preview}")
        print(f"  (assistant length: {len(messages_raw[-1]['content'])} chars)")

        messages = [{"role": "system", "content": MATH_SYSTEM_PROMPT}]
        for m in messages_raw:
            messages.append({"role": m["role"], "content": m["content"]})

        try:
            datum = conversation_to_datum(messages, renderer, common_config.max_length, train_on_what)
            datums.append(datum)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    print(f"\nMath: {len(datums)}/{SMOKE_EXAMPLES} examples converted to datums")
    return datums


def build_code_data(renderer, common_config):
    """Load 10 code examples, print them, return as dataset."""
    print(f"\n{'='*60}")
    print("CODE DATA (KodCode-V1-SFT-R1)")
    print(f"{'='*60}")

    ds = datasets.load_dataset("KodCode/KodCode-V1-SFT-R1", split="train", streaming=True)
    ds = cast(datasets.IterableDataset, ds)

    train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    datums = []
    for i, row in enumerate(ds):
        if i >= SMOKE_EXAMPLES:
            break

        conversations = row.get("conversations", [])
        user_content = ""
        assistant_content = ""
        for turn in conversations:
            if turn["from"] == "human":
                user_content = turn["value"]
            elif turn["from"] == "gpt":
                assistant_content = turn["value"]

        # Print the example
        print(f"\n--- Code Example {i+1} ---")
        preview_u = user_content[:200] + "..." if len(user_content) > 200 else user_content
        preview_a = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
        print(f"  [user]: {preview_u}")
        print(f"  [assistant]: {preview_a}")
        print(f"  (assistant length: {len(assistant_content)} chars)")

        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        try:
            datum = conversation_to_datum(messages, renderer, common_config.max_length, train_on_what)
            datums.append(datum)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    print(f"\nCode: {len(datums)}/{SMOKE_EXAMPLES} examples converted to datums")
    return datums


async def run_smoke_train(task_name: str, datums: list[tinker.Datum]):
    """Run 1 training step on the given datums."""
    print(f"\n{'='*60}")
    print(f"TRAINING SMOKE TEST: {task_name}")
    print(f"  Model: {BASE_MODEL_NAME}")
    print(f"  Examples: {len(datums)}")
    print(f"  Steps: 1")
    print(f"{'='*60}")

    batch_size = len(datums)
    train_dataset = FixedListDataset(datums, batch_size=batch_size)
    test_dataset = FixedListDataset(datums[:2], batch_size=2)

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)

    @chz.chz
    class SmokeBuilder(ChatDatasetBuilder):
        def __call__(self):
            return train_dataset, test_dataset

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_TOKENS,
        batch_size=batch_size,
    )

    dataset_builder = SmokeBuilder(common_config=common_config)

    log_path = f"{TRAINING_BASE}/base_smoke_{task_name}"

    config = train.Config(
        log_path=log_path,
        model_name=BASE_MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=2e-4,
        lr_schedule="cosine",
        num_epochs=1,
        lora_rank=LORA_RANK,
        save_every=0,
        eval_every=1,
    )

    await train.main(config)
    print(f"\n{task_name} smoke test complete!")


async def main():
    print(f"Base Model Distillation Smoke Test")
    print(f"Model: {BASE_MODEL_NAME}")
    print(f"Examples per task: {SMOKE_EXAMPLES}")

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook import renderers

    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_TOKENS,
        batch_size=SMOKE_EXAMPLES,
    )

    # Load and print data
    math_datums = build_math_data(renderer, common_config)
    code_datums = build_code_data(renderer, common_config)

    # Run training smoke tests
    await run_smoke_train("math", math_datums)
    await run_smoke_train("code", code_datums)

    print(f"\n{'='*60}")
    print("ALL SMOKE TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

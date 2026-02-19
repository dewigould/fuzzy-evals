"""Launch formatting SFT runs: teach cross-domain format to distilled checkpoints.

- Math-distilled → train on code.jsonl (learn ```python``` format)
- Code-distilled → train on math.jsonl (learn \boxed{} format)

1 epoch, 19 examples each, batch_size=19 (1 step).
"""

import asyncio
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, "/workspace/tinker-cookbook")

from tinker_cookbook import model_info  # for get_recommended_renderer_name
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
import chz
import json
import tinker

from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT

FORMAT_DATA_DIR = EXPERIMENT_DIR / "formatting_training_data"
TRAINING_DIR = EXPERIMENT_DIR / "training_runs"

# 4 runs: (name, load_checkpoint_state_path, data_file, system_prompt, base_model)
RUNS = [
    # Base math s300 → learn code format
    (
        "sonnet_math_qwen_4k_step300_fmt_code",
        "tinker://f889cb10-1497-5573-951c-c538b60bb629:train:0/weights/000300",
        FORMAT_DATA_DIR / "code.jsonl",
        CODE_SYSTEM_PROMPT,
        "Qwen/Qwen3-30B-A3B-Base",
    ),
    # Base code s200 → learn math format
    (
        "sonnet_code_qwen_4k_step200_fmt_math",
        "tinker://f0452a6d-df68-5a5d-b3ea-c4c486173024:train:0/weights/000200",
        FORMAT_DATA_DIR / "math.jsonl",
        MATH_SYSTEM_PROMPT,
        "Qwen/Qwen3-30B-A3B-Base",
    ),
    # Instruct math s200 → learn code format
    (
        "sonnet_math_qwen_instruct_4k_step200_fmt_code",
        "tinker://137bc595-33d9-53be-a403-0a74affd0405:train:0/weights/000200",
        FORMAT_DATA_DIR / "code.jsonl",
        CODE_SYSTEM_PROMPT,
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
    # Instruct code final → learn math format
    (
        "sonnet_code_qwen_instruct_4k_final_fmt_math",
        "tinker://9c09ed93-3808-5b9c-912e-418bf80a3507:train:0/weights/final",
        FORMAT_DATA_DIR / "math.jsonl",
        MATH_SYSTEM_PROMPT,
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
]

BATCH_SIZE = 19  # all examples in one batch
LR = 1e-4
LORA_RANK = 32
MAX_LENGTH = 4096


class FixedListDataset(SupervisedDataset):
    def __init__(self, data, batch_size):
        self._data = data
        self._batch_size = batch_size

    def __len__(self):
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def get_batch(self, idx):
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._data))
        return self._data[start:end]


@chz.chz
class FormatDataBuilder(ChatDatasetBuilder):
    data_path: str = ""
    system_prompt: str = ""

    def __call__(self):
        with open(self.data_path) as f:
            rows = [json.loads(line) for line in f]

        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        datums = []
        for row in rows:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(row["messages"])
            datums.append(conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            ))

        print(f"  Loaded {len(datums)} formatting examples from {self.data_path}")
        return FixedListDataset(datums, self.common_config.batch_size), None


async def run_one(name, checkpoint_path, data_path, system_prompt, base_model):
    print(f"\n{'='*60}")
    print(f"Format SFT: {name}")
    print(f"  Load from: {checkpoint_path}")
    print(f"  Data: {data_path} ({sum(1 for _ in open(data_path))} examples)")
    print(f"  Base model: {base_model}")
    print(f"{'='*60}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=model_info.get_recommended_renderer_name(base_model),
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    dataset_builder = FormatDataBuilder(
        common_config=common_config,
        data_path=str(data_path),
        system_prompt=system_prompt,
    )

    config = train.Config(
        log_path=str(TRAINING_DIR / name),
        model_name=base_model,
        load_checkpoint_path=checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=LR,
        lr_schedule="constant",
        num_epochs=1,
        lora_rank=LORA_RANK,
        save_every=1,
        eval_every=1,
    )

    await train.main(config)
    print(f"\nDone: {name}")


async def main():
    for name, ckpt, data, prompt, model in RUNS:
        await run_one(name, ckpt, data, prompt, model)


if __name__ == "__main__":
    asyncio.run(main())

"""
Experiment 2b: Code SFT — 500 steps of SFT on NVIDIA/OpenCodeReasoning.
Uses streaming HF dataset with batch_size=50 and max_prompts=25000.
Trains on the full reasoning output (thinking + solution).
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
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/workspace/results_06_02_v2/training/code_sft'
BATCH_SIZE = 50
MAX_PROMPTS = 25000  # 500 steps * 50 batch_size


@chz.chz
class OpenCodeReasoningBuilder(ChatDatasetBuilder):
    """SFT dataset builder for NVIDIA/OpenCodeReasoning with streaming."""
    buffer_size: int = 50_000
    max_prompts: int = MAX_PROMPTS

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load both splits and interleave
        ds0 = datasets.load_dataset(
            "nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True
        )
        ds1 = datasets.load_dataset(
            "nvidia/OpenCodeReasoning", "split_1", split="split_1", streaming=True
        )
        ds = datasets.interleave_datasets([cast(datasets.IterableDataset, ds0),
                                            cast(datasets.IterableDataset, ds1)])
        ds = cast(datasets.IterableDataset, ds)

        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        def map_fn(row: dict) -> tinker.Datum:
            # Schema: input=problem, output=thinking+reasoning, solution=code
            # Use 'output' as the full reasoning trace (includes <think> + answer)
            problem = row.get("input", "")
            # Combine reasoning trace and clean solution
            reasoning = row.get("output", "")
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": reasoning},
            ]
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=ds,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,
            map_fn=map_fn,
            buffer_size=self.buffer_size,
        )
        return train_dataset, None


def main():
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=BATCH_SIZE,
    )

    dataset_builder = OpenCodeReasoningBuilder(
        common_config=common_config,
        max_prompts=MAX_PROMPTS,
    )

    config = train.Config(
        log_path=LOG_PATH,
        model_name=MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=32,
        save_every=50,
        eval_every=50,
    )

    asyncio.run(train.main(config))


if __name__ == '__main__':
    main()

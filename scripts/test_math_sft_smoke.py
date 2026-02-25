"""
Smoke test for Math SFT on OpenR1-Math-220k reasoning traces.

Quick validation (2 steps):
1. Load 100 examples, split 90/10
2. Run 2 SFT steps with batch_size=10, max_length=8192
3. Verify: NLL decreases, model generates <think> format
"""

import asyncio
import json
import sys
from typing import cast

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

import chz
import datasets
import tinker
from tinker_cookbook import model_info, renderers
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
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/workspace/results_10_02/math_sft/smoke_test'
BATCH_SIZE = 10
MAX_PROMPTS = 90  # 90 train examples
TEST_SIZE = 10
MAX_LENGTH = 8192


class FixedListDataset(SupervisedDataset):
    """Tiny in-memory dataset for test split."""

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
class SmokeMathSFTBuilder(ChatDatasetBuilder):
    """SFT dataset builder for smoke test with small train/test split."""
    max_prompts: int = MAX_PROMPTS

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        print(f"Loading OpenR1-Math-220k (first {self.max_prompts + TEST_SIZE} examples)...")
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

        # Collect train + test examples from stream
        all_rows = []
        for row in ds:
            all_rows.append(row)
            if len(all_rows) >= self.max_prompts + TEST_SIZE:
                break

        print(f"  Collected {len(all_rows)} examples")

        # Split
        train_rows = all_rows[:self.max_prompts]
        test_rows = all_rows[self.max_prompts:]

        # Convert test rows to datums for NLL evaluator
        test_datums = []
        for row in test_rows:
            try:
                datum = map_fn(row)
                test_datums.append(datum)
            except Exception as e:
                print(f"  Skipping test example: {e}")
        print(f"  Test set: {len(test_datums)} datums")

        # Show a sample to verify format
        if all_rows:
            sample_msg = all_rows[0].get("messages", [])
            if sample_msg and len(sample_msg) >= 2:
                assistant_content = sample_msg[-1].get("content", "")
                print(f"\n  Sample assistant message (first 200 chars):")
                print(f"  {assistant_content[:200]}")
                has_think = "<think>" in assistant_content
                has_think_close = "</think>" in assistant_content
                print(f"  Has <think>: {has_think}, Has </think>: {has_think_close}")

        # Build streaming train dataset
        # Re-create iterable from collected rows for streaming dataset
        def row_generator():
            for row in train_rows:
                yield row

        train_iter = datasets.IterableDataset.from_generator(row_generator)

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=train_iter,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,
            map_fn=map_fn,
            buffer_size=200,
        )

        test_dataset = FixedListDataset(test_datums, batch_size=TEST_SIZE)
        return train_dataset, test_dataset


async def verify_generation(model_name, log_path):
    """After training, sample from the model with <think>\\n prefix to verify format."""
    print("\n" + "=" * 60)
    print("Post-training generation check")
    print("=" * 60)

    from tinker_cookbook import checkpoint_utils

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sc = tinker.ServiceClient()

    # Load the last checkpoint
    try:
        ckpt = checkpoint_utils.get_last_checkpoint(log_path)
        sampler_path = ckpt['sampler_path']
        print(f"Loaded checkpoint: {sampler_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Using base model instead...")
        sampler_path = None

    if sampler_path:
        sampling_client = sc.create_sampling_client(model_path=sampler_path)
    else:
        sampling_client = sc.create_sampling_client(base_model=model_name)

    # Test prompt
    prompt = "Solve the following math problem. Put your final answer in \\boxed{}.\n\nWhat is 2 + 3?"
    messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)

    # Seed with <think>\n
    prefix_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)
    for token_id in prefix_tokens:
        model_input.append_int(token_id)

    response = await sampling_client.sample_async(
        model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=renderer.get_stop_sequences(),
        ),
    )

    parsed, _ = renderer.parse_response(response.sequences[0].tokens)
    content = "<think>\n" + parsed['content']

    print(f"\nPrompt: {prompt}")
    print(f"\nModel output (first 500 chars):")
    print(content[:500])
    print(f"\nOutput length: {len(content)} chars")
    print(f"Contains <think>: {'<think>' in content}")
    print(f"Contains </think>: {'</think>' in content}")
    print(f"Contains \\boxed: {'\\boxed' in content or 'boxed' in content}")

    return content


def main():
    print("=" * 60)
    print("Math SFT Smoke Test")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Train examples: {MAX_PROMPTS}")
    print(f"  Test examples: {TEST_SIZE}")
    print(f"  Steps: {MAX_PROMPTS // BATCH_SIZE} (= {MAX_PROMPTS} / {BATCH_SIZE})")
    print(f"  Log path: {LOG_PATH}")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    dataset_builder = SmokeMathSFTBuilder(
        common_config=common_config,
        max_prompts=MAX_PROMPTS,
    )

    config = train.Config(
        log_path=LOG_PATH,
        model_name=MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        lr_schedule="constant",
        num_epochs=1,
        lora_rank=32,
        save_every=0,  # Don't save checkpoints for smoke test
        eval_every=1,  # Eval every step to see NLL trend
    )

    print("\nStarting training...")
    asyncio.run(train.main(config))

    # Check metrics
    import os
    metrics_file = os.path.join(LOG_PATH, 'metrics.jsonl')
    if os.path.exists(metrics_file):
        print("\n" + "=" * 60)
        print("Training Metrics")
        print("=" * 60)
        nlls = []
        with open(metrics_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    step = d.get('progress/batch', '?')
                    train_nll = d.get('train_mean_nll', None)
                    test_nll = d.get('test/nll', None)
                    if train_nll is not None or test_nll is not None:
                        parts = [f"Step {step}"]
                        if train_nll is not None:
                            parts.append(f"train_nll={train_nll:.4f}")
                        if test_nll is not None:
                            parts.append(f"test_nll={test_nll:.4f}")
                            nlls.append(test_nll)
                        print(f"  {' | '.join(parts)}")
                except json.JSONDecodeError:
                    continue

        if len(nlls) >= 2:
            if nlls[-1] < nlls[0]:
                print(f"\n  NLL decreased: {nlls[0]:.4f} -> {nlls[-1]:.4f} (PASS)")
            else:
                print(f"\n  NLL did NOT decrease: {nlls[0]:.4f} -> {nlls[-1]:.4f} (WARN)")
        elif nlls:
            print(f"\n  Only {len(nlls)} NLL measurement (need 2+ for trend)")
    else:
        print(f"\nWARNING: No metrics file found at {metrics_file}")

    # Verify generation
    print("\nVerifying generation with <think> seeding...")
    asyncio.run(verify_generation(MODEL_NAME, LOG_PATH))

    print("\n" + "=" * 60)
    print("Smoke test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

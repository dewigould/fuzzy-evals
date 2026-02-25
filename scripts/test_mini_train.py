"""
Quick 2-step training run to verify the code RLVR pipeline works end-to-end
with all the performance fixes applied.
"""
import asyncio
import sys
import time

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from run_code_rlvr import (
    CodeProblemEnv, SimpleCodeDataset, load_deepcoder_tasks_streaming,
    HARD_CAP_SECONDS, TRAIN_TIMEOUT,
)
from run_code_rlvr_single import CodeDatasetBuilderWithTest, TruncatedDataset, TruncatedDatasetBuilder
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main as train_main

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/tmp/test_mini_code_train'

async def main():
    print("=" * 60)
    print("Mini Code RLVR Training Test (2 steps)")
    print(f"  HARD_CAP_SECONDS={HARD_CAP_SECONDS}")
    print(f"  TRAIN_TIMEOUT={TRAIN_TIMEOUT}")
    print("=" * 60)

    # Use small batch size and group size for quick test
    inner_builder = CodeDatasetBuilderWithTest(
        model_name=MODEL_NAME,
        batch_size=4,   # small batch
        group_size=2,   # small group = 4*2=8 envs per batch
        format_coef=0.1,
        timeout=TRAIN_TIMEOUT,
        seed=42,
    )

    dataset_builder = TruncatedDatasetBuilder(inner_builder, max_steps=2)

    config = Config(
        learning_rate=1e-5,
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        lora_rank=32,
        max_tokens=2048,
        temperature=1.0,
        log_path=LOG_PATH,
        save_every=999,  # don't save
        eval_every=999,  # don't eval
        kl_penalty_coef=0.02,
        kl_reference_config=KLReferenceConfig(base_model=MODEL_NAME),
    )

    t0 = time.time()
    await train_main(config)
    elapsed = time.time() - t0
    print(f"\nTotal time for 2 steps: {elapsed:.1f}s")
    print(f"Average time per step: {elapsed/2:.1f}s")

if __name__ == '__main__':
    asyncio.run(main())

"""
Experiment 1a: Math RLVR — 500 steps of RLVR on Hendrycks MATH.
Uses MathDatasetBuilder with batch_size=24 to get 500 batches from ~12k problems.
"""

import asyncio
import sys
from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder
from tinker_cookbook.rl.train import Config, main as train_main
from tinker_cookbook import model_info

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/workspace/results_06_02_v2/training/math_rlvr'


def main():
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    dataset_builder = get_math_dataset_builder(
        dataset_name="math",
        batch_size=24,       # ~12000/24 = 500 batches
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        group_size=4,
        seed=42,
    )

    config = Config(
        learning_rate=1e-5,
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        lora_rank=32,
        max_tokens=512,
        temperature=1.0,
        log_path=LOG_PATH,
        save_every=50,
        eval_every=50,
    )

    asyncio.run(train_main(config))


if __name__ == '__main__':
    main()

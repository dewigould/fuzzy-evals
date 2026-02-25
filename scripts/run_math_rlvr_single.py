"""
Run a single Math RLVR training config for the hyperparameter sweep.
Usage: python run_math_rlvr_single.py <config_name> --max_tokens <value>

Each config runs for 75 steps with eval every 25 steps on MATH-500.
Results are saved to /workspace/results_10_02/<config_name>/
"""

import argparse
import asyncio
import sys
from typing import Sequence

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main as train_main
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook import model_info

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_STEPS = 75
EVAL_EVERY = 25
SAVE_EVERY = 25
LORA_RANK = 32
TEMPERATURE = 1.0

SWEEP_CONFIGS = {
    "A_moderate": {
        "learning_rate": 1e-5,
        "kl_penalty_coef": 0.05,
        "group_size": 8,
        "batch_size": 24,
        "loss_fn": "importance_sampling",
        "remove_constant_reward_groups": True,
    },
    "B_conservative": {
        "learning_rate": 5e-6,
        "kl_penalty_coef": 0.05,
        "group_size": 8,
        "batch_size": 24,
        "loss_fn": "importance_sampling",
        "remove_constant_reward_groups": True,
    },
    "C_aggressive": {
        "learning_rate": 3e-5,
        "kl_penalty_coef": 0.1,
        "group_size": 8,
        "batch_size": 24,
        "loss_fn": "importance_sampling",
        "remove_constant_reward_groups": True,
    },
    "D_baseline_v2": {
        "learning_rate": 1e-5,
        "kl_penalty_coef": 0.02,
        "group_size": 4,
        "batch_size": 48,
        "loss_fn": "importance_sampling",
        "remove_constant_reward_groups": False,
    },
    "E_ppo_moderate": {
        "learning_rate": 1e-5,
        "kl_penalty_coef": 0.05,
        "group_size": 8,
        "batch_size": 24,
        "loss_fn": "ppo",
        "remove_constant_reward_groups": True,
    },
    "F_large_group": {
        "learning_rate": 1e-5,
        "kl_penalty_coef": 0.05,
        "group_size": 16,
        "batch_size": 12,
        "loss_fn": "importance_sampling",
        "remove_constant_reward_groups": True,
    },
}


class TruncatedDataset(RLDataset):
    """Wraps an RLDataset to limit the number of batches (= training steps)."""

    def __init__(self, inner: RLDataset, max_batches: int):
        self.inner = inner
        self.max_batches = max_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.inner.get_batch(index)

    def __len__(self) -> int:
        return min(len(self.inner), self.max_batches)


class TruncatedDatasetBuilder:
    """Wraps an RLDatasetBuilder to truncate the training dataset.
    Not a chz dataclass since we need mutable __init__."""

    def __init__(self, inner_builder: RLDatasetBuilder, max_steps: int):
        self._inner_builder = inner_builder
        self._max_steps = max_steps

    async def __call__(self):
        train_ds, test_ds = await self._inner_builder()
        return TruncatedDataset(train_ds, self._max_steps), test_ds


def main():
    parser = argparse.ArgumentParser(description='Run a single RLVR sweep config')
    parser.add_argument('config_name', choices=list(SWEEP_CONFIGS.keys()),
                        help='Config name to run')
    parser.add_argument('--max_tokens', type=int, required=True,
                        help='Max tokens for generation (from calibration)')
    args = parser.parse_args()

    cfg_params = SWEEP_CONFIGS[args.config_name]
    log_path = f'/workspace/results_10_02/{args.config_name}'

    print(f"{'='*60}")
    print(f"RLVR Sweep: {args.config_name}")
    print(f"  LR={cfg_params['learning_rate']}, KL={cfg_params['kl_penalty_coef']}, "
          f"group_size={cfg_params['group_size']}, batch_size={cfg_params['batch_size']}")
    print(f"  loss_fn={cfg_params['loss_fn']}, remove_constant={cfg_params['remove_constant_reward_groups']}")
    print(f"  max_tokens={args.max_tokens}, steps={MAX_STEPS}")
    print(f"  log_path={log_path}")
    print(f"{'='*60}")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    # Build the inner dataset builder
    inner_builder = get_math_dataset_builder(
        dataset_name="math",
        batch_size=cfg_params['batch_size'],
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        group_size=cfg_params['group_size'],
        seed=42,
    )

    # Wrap to truncate to MAX_STEPS
    dataset_builder = TruncatedDatasetBuilder(inner_builder, MAX_STEPS)

    config = Config(
        learning_rate=cfg_params['learning_rate'],
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        lora_rank=LORA_RANK,
        max_tokens=args.max_tokens,
        temperature=TEMPERATURE,
        log_path=log_path,
        save_every=SAVE_EVERY,
        eval_every=EVAL_EVERY,
        kl_penalty_coef=cfg_params['kl_penalty_coef'],
        kl_reference_config=KLReferenceConfig(base_model=MODEL_NAME),
        loss_fn=cfg_params['loss_fn'],
        remove_constant_reward_groups=cfg_params['remove_constant_reward_groups'],
    )

    asyncio.run(train_main(config))


if __name__ == '__main__':
    main()

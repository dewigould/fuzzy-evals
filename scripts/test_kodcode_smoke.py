"""
Smoke test: Run 2 steps of KodCode RLVR to verify the full pipeline works
before launching the 6-config sweep.
"""
import asyncio
import sys
import time

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main as train_main
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from run_code_rlvr import (
    KodCodeProblemEnv,
    SimpleCodeDataset,
    load_kodcode_tasks,
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


class SmokeTestDatasetBuilder:
    """Tiny dataset for smoke testing: 20 training problems, 5 test problems."""

    def __init__(self):
        pass

    async def __call__(self):
        tokenizer = get_tokenizer(MODEL_NAME)
        renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        # Load a small number of tasks
        all_tasks = load_kodcode_tasks(difficulty="easy", max_tasks=7000, seed=42)

        # Use first 20 for training, next 5 for test
        train_tasks = all_tasks[:20]
        test_tasks = all_tasks[20:25]

        train_builders = []
        for task in train_tasks:
            def make_env(t=task):
                return KodCodeProblemEnv(
                    problem=t.question,
                    test_code=t.test_code,
                    renderer=renderer,
                    format_coef=0.1,
                    timeout=10,
                )
            train_builders.append(ProblemGroupBuilder(
                env_thunk=make_env,
                num_envs=4,
                dataset_name="code",
            ))

        test_builders = []
        for task in test_tasks:
            def make_test_env(t=task):
                return KodCodeProblemEnv(
                    problem=t.question,
                    test_code=t.test_code,
                    renderer=renderer,
                    format_coef=0.1,
                    timeout=10,
                )
            test_builders.append(ProblemGroupBuilder(
                env_thunk=make_test_env,
                num_envs=1,
                dataset_name="code",
            ))

        # batch_size=10 → 2 batches for 20 problems
        train_ds = SimpleCodeDataset(train_builders, 10)
        test_ds = SimpleCodeDataset(test_builders, len(test_builders))

        return train_ds, test_ds


async def main():
    print("=" * 60)
    print("SMOKE TEST: KodCode RLVR (2 steps)")
    print("=" * 60)

    t0 = time.time()

    config = Config(
        learning_rate=1e-5,
        dataset_builder=SmokeTestDatasetBuilder(),
        model_name=MODEL_NAME,
        lora_rank=32,
        max_tokens=1024,
        temperature=1.0,
        log_path='/workspace/results_10_02/code_smoke_test',
        save_every=100,  # Don't save during smoke test
        eval_every=1,    # Eval after each step
        kl_penalty_coef=0.02,
        kl_reference_config=KLReferenceConfig(base_model=MODEL_NAME),
    )

    await train_main(config)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SMOKE TEST COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())

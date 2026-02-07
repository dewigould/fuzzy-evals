"""
Experiment 1b: Code RLVR — 500 steps on coding problems (Deepcoder/TACO/LCB).

Uses ProblemEnv (single-turn: generate code → check tests → reward) like math RLVR.
No tool calling needed. Streaming dataset, subprocess code execution.
"""

import asyncio
import gc
import json
import os
import random
import re
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

from dataclasses import dataclass
from typing import Sequence
from datasets import load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.code_rl.code_env import (
    DeepcoderTask,
    _build_question,
    _ensure_dict,
    _normalize_tests,
)
from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.train import Config, main as train_main
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/workspace/results_06_02_v2/training/code_rlvr'
MAX_TASKS = 5000
BATCH_SIZE = 10  # 5000/10 = 500 batches


# ============================================================
# Code Problem Environment (single-turn, no tool calling)
# ============================================================

def extract_code(text):
    """Extract Python code from model response."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


def run_code_tests_sync(code, tests, timeout=6):
    """Run code against test cases using subprocess."""
    test_cases = postprocess_lcb_sample(tests)
    test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
    total_timeout = (timeout + 1) * test_cnt + 5

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test_cases.txt"), "w") as f:
            f.write(json.dumps(test_cases))
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "testing_util.py"), "w") as f:
            f.write(TEST_UTIL)
        with open(os.path.join(tmpdir, "run.py"), "w") as f:
            f.write(TEST_CODE % {"timeout": timeout})

        try:
            result = subprocess.run(
                [sys.executable, "run.py"],
                cwd=tmpdir,
                capture_output=True,
                timeout=total_timeout,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


class CodeProblemEnv(ProblemEnv):
    """Single-turn code RL environment. Model generates code, we check tests."""

    def __init__(self, problem, tests, renderer, format_coef=0.1, timeout=6):
        super().__init__(renderer, format_coef=format_coef)
        self.problem = problem
        self.tests = tests
        self.timeout = timeout

    def get_question(self):
        return (self.problem +
                "\n\nProvide your solution in a ```python code block.")

    def check_format(self, sample_str):
        return extract_code(sample_str) is not None

    def check_answer(self, sample_str):
        code = extract_code(sample_str)
        if code is None:
            return False
        return run_code_tests_sync(code, self.tests, timeout=self.timeout)

    def get_reference_answer(self):
        return "(code solution)"


# ============================================================
# Dataset Builder using ProblemGroupBuilder
# ============================================================

class SimpleCodeDataset(RLDataset):
    def __init__(self, builders, batch_size):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        return self.builders[start:end]

    def __len__(self):
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@dataclass
class SimpleCodeDatasetBuilder:
    model_name: str
    batch_size: int
    group_size: int
    format_coef: float = 0.1
    timeout: int = 6
    seed: int = 42

    async def __call__(self):
        # Load tasks
        tasks = load_deepcoder_tasks_streaming("train", seed=self.seed)

        # Set up renderer
        tokenizer = get_tokenizer(self.model_name)
        renderer_name = model_info.get_recommended_renderer_name(self.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        # Build ProblemGroupBuilders (like math RLVR)
        builders = []
        for task in tasks:
            def make_env(t=task):
                return CodeProblemEnv(
                    problem=t.problem,
                    tests=t.tests,
                    renderer=renderer,
                    format_coef=self.format_coef,
                    timeout=self.timeout,
                )
            builders.append(ProblemGroupBuilder(
                env_thunk=make_env,
                num_envs=self.group_size,
                dataset_name="code",
            ))

        dataset = SimpleCodeDataset(builders, self.batch_size)
        return dataset, None  # No test dataset


# ============================================================
# Streaming dataset loader
# ============================================================

def load_deepcoder_tasks_streaming(split="train", seed=0):
    """Load DeepCoder tasks via HF streaming (no disk caching)."""
    if split == "test":
        return []
    print(f"Loading DeepCoder tasks (streaming), split={split}, max={MAX_TASKS}")

    names = ("primeintellect", "taco", "lcbv5")
    tasks = []
    for name in names:
        print(f"  Streaming {name}...")
        try:
            ds = load_dataset(
                "agentica-org/DeepCoder-Preview-Dataset",
                name=name, split=split, streaming=True,
            )
            count = 0
            for row in ds:
                metadata = _ensure_dict(row.get("metadata", {}))
                raw_tests = row.get("tests") or row.get("ground_truth")
                tests = _normalize_tests(raw_tests, metadata)
                if not tests:
                    continue
                problem = _build_question(row)
                if problem is None:
                    continue
                starter_code = row.get("starter_code")
                if isinstance(starter_code, str) and not starter_code.strip():
                    starter_code = None
                tasks.append(DeepcoderTask(
                    problem=problem,
                    tests=tests,
                    starter_code=starter_code if isinstance(starter_code, str) else None,
                ))
                count += 1
                if count % 2000 == 0:
                    print(f"    Loaded {count} tasks from {name}...")
                if len(tasks) >= MAX_TASKS:
                    break
            del ds
            gc.collect()
        except Exception as e:
            print(f"  Warning: Failed to load {name}: {e}")
            continue
        if len(tasks) >= MAX_TASKS:
            break

    print(f"  Total tasks loaded: {len(tasks)}")
    gc.collect()
    if split == "train":
        random.Random(seed).shuffle(tasks)
    return tasks


# ============================================================
# Main
# ============================================================

def main():
    print("Code RLVR: ProblemEnv (single-turn, no tool calling)")
    print(f"Streaming dataset (max {MAX_TASKS} tasks, batch_size={BATCH_SIZE})")

    dataset_builder = SimpleCodeDatasetBuilder(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        group_size=4,
        format_coef=0.1,
        timeout=6,
        seed=42,
    )

    config = Config(
        learning_rate=1e-5,
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        lora_rank=32,
        max_tokens=4096,
        temperature=1.0,
        log_path=LOG_PATH,
        save_every=50,
        eval_every=50,
    )

    asyncio.run(train_main(config))


if __name__ == '__main__':
    main()

"""
Experiment 1b: Code RLVR — training on coding problems.

Supports two dataset backends:
- DeepCoder (original, competitive programming — 0% solve rate for 8B)
- KodCode (clean pytest-style tests — ~30% solve rate for 8B)

Uses ProblemEnv (single-turn: generate code → check tests → reward) like math RLVR.
No tool calling needed. Streaming dataset, subprocess code execution.
"""

import asyncio
import concurrent.futures
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

import tinker
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
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main as train_main
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = '/workspace/results_06_02_v2/training/code_rlvr'
MAX_TASKS = 5000
BATCH_SIZE = 20  # 5000/20 = 250 batches

# --- Training-time test execution parameters ---
# Hard wall-clock cap in seconds, independent of test count
HARD_CAP_SECONDS = 30
# Per-test timeout for training
TRAIN_TIMEOUT = 4
# Thread pool for non-blocking subprocess execution
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


# ============================================================
# Code Problem Environment (single-turn, no tool calling)
# ============================================================

def extract_code(text):
    """Extract Python code from model response."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


def run_code_tests_sync(code, tests, timeout=6, hard_cap=None):
    """Run code against test cases using subprocess with memory limits.

    Args:
        code: Python code to test
        tests: Test cases in DeepCoder format
        timeout: Per-test timeout in seconds
        hard_cap: If set, hard wall-clock cap in seconds
    """
    test_cases = postprocess_lcb_sample(tests)
    test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
    total_timeout = (timeout + 1) * test_cnt + 5

    # Apply hard wall-clock cap
    if hard_cap is not None:
        total_timeout = min(total_timeout, hard_cap)

    # Fix 3: Removed signal.alarm — it gets overwritten by per-test alarms
    # in testing_util.py, making it ineffective. subprocess.run timeout is
    # the real outer guard.
    # Fix 6: Increased RLIMIT_AS from 512MB to 2GB — the test harness imports
    # numpy which needs >512MB, causing OOM kills and 20s hangs.
    resource_wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({total_timeout}, {total_timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test_cases.txt"), "w") as f:
            f.write(json.dumps(test_cases))
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "testing_util.py"), "w") as f:
            f.write(TEST_UTIL)
        with open(os.path.join(tmpdir, "run.py"), "w") as f:
            f.write(resource_wrapper + (TEST_CODE % {"timeout": timeout}))

        try:
            result = subprocess.run(
                [sys.executable, "run.py"],
                cwd=tmpdir,
                capture_output=True,
                timeout=total_timeout + 2,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def run_kodcode_tests_sync(code, test_code, timeout=10):
    """Run code against KodCode pytest-style tests.

    Writes the model's code as solution.py, the test as test_solution.py,
    and runs a test runner in a subprocess with resource limits.
    """
    resource_wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))\n"
    )

    runner = resource_wrapper + '''
import sys
import importlib.util
import traceback

sys.path.insert(0, ".")

try:
    import solution

    spec = importlib.util.spec_from_file_location("test_solution", "test_solution.py")
    test_mod = importlib.util.module_from_spec(spec)

    # Inject solution functions for tests that call functions directly
    for name in dir(solution):
        if not name.startswith('_'):
            setattr(test_mod, name, getattr(solution, name))

    spec.loader.exec_module(test_mod)

    test_funcs = [name for name in dir(test_mod) if name.startswith('test_')]
    failed = 0
    for name in test_funcs:
        try:
            getattr(test_mod, name)()
        except Exception:
            failed += 1

    if failed > 0 or len(test_funcs) == 0:
        sys.exit(1)
    else:
        sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(test_code)
        with open(os.path.join(tmpdir, "run_tests.py"), "w") as f:
            f.write(runner)

        try:
            result = subprocess.run(
                [sys.executable, "run_tests.py"],
                cwd=tmpdir,
                capture_output=True,
                timeout=timeout + 2,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


class CodeProblemEnv(ProblemEnv):
    """Single-turn code RL environment. Model generates code, we check tests.

    Overrides step() to run check_answer in a thread pool (Fix 5),
    preventing blocking of the asyncio event loop during concurrent rollouts.
    """

    def __init__(self, problem, tests, renderer, format_coef=0.1, timeout=TRAIN_TIMEOUT,
                 hard_cap=HARD_CAP_SECONDS):
        super().__init__(renderer, format_coef=format_coef)
        self.problem = problem
        self.tests = tests
        self.timeout = timeout
        self.hard_cap = hard_cap

    def get_question(self):
        return (self.problem +
                "\n\nProvide your solution in a ```python code block.")

    def check_format(self, sample_str):
        return extract_code(sample_str) is not None

    def check_answer(self, sample_str):
        code = extract_code(sample_str)
        if code is None:
            return False
        return run_code_tests_sync(
            code, self.tests,
            timeout=self.timeout,
            hard_cap=self.hard_cap,
        )

    def get_reference_answer(self):
        return "(code solution)"

    async def step(self, action: Action) -> StepResult:
        """Override step() to run blocking check_answer in a thread pool (Fix 5)."""
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))

        # Fix 5: Run blocking subprocess in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        correct_answer = float(await loop.run_in_executor(
            _test_executor, self.check_answer, content
        ))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content'][:200]}...")
        logtree.log_text(
            f"Format Valid: {'Y' if correct_format else 'N'}, Correct: {'Y' if correct_answer else 'N'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


class KodCodeProblemEnv(ProblemEnv):
    """Single-turn code RL environment using KodCode pytest-style tests."""

    def __init__(self, problem, test_code, renderer, format_coef=0.1, timeout=10):
        super().__init__(renderer, format_coef=format_coef)
        self.problem = problem
        self.test_code = test_code
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
        return run_kodcode_tests_sync(code, self.test_code, timeout=self.timeout)

    def get_reference_answer(self):
        return "(code solution)"

    async def step(self, action: Action) -> StepResult:
        """Override step() to run blocking check_answer in a thread pool."""
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))

        loop = asyncio.get_event_loop()
        correct_answer = float(await loop.run_in_executor(
            _test_executor, self.check_answer, content
        ))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content'][:200]}...")
        logtree.log_text(
            f"Format Valid: {'Y' if correct_format else 'N'}, Correct: {'Y' if correct_answer else 'N'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


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
# KodCode dataset loader
# ============================================================

@dataclass
class KodCodeTask:
    """A single KodCode problem."""
    question: str
    test_code: str
    gpt_difficulty: str
    gpt_pass_pct: float

def load_kodcode_tasks(difficulty="easy", max_tasks=5000, seed=42):
    """Load KodCode tasks, filtered by difficulty.

    Only includes problems with:
    - from-solution-import style tests
    - No numpy/pandas/torch dependencies in tests
    """
    print(f"Loading KodCode tasks (difficulty={difficulty}, max={max_tasks})")
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    tasks = []
    for row in ds:
        if row['gpt_difficulty'] != difficulty:
            continue
        test = row['test']
        if 'from solution import' not in test:
            continue
        test_lower = test.lower()
        if any(lib in test_lower for lib in ['import numpy', 'import pandas', 'import torch']):
            continue
        tasks.append(KodCodeTask(
            question=row['question'],
            test_code=test,
            gpt_difficulty=row['gpt_difficulty'],
            gpt_pass_pct=row['gpt_pass_percentage'],
        ))
        if len(tasks) >= max_tasks:
            break

    print(f"  Loaded {len(tasks)} KodCode tasks")
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
        kl_penalty_coef=0.02,
        kl_reference_config=KLReferenceConfig(base_model=MODEL_NAME),
    )

    asyncio.run(train_main(config))


if __name__ == '__main__':
    main()

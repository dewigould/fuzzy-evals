"""Pairwise effort judging — configuration constants."""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/workspace/.env")

# -- API --
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# -- Judge ensemble --
JUDGE_MODELS = [
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-pro",
]

JUDGE_SHORT_NAMES = {
    "openai/gpt-5.2": "gpt52",
    "anthropic/claude-sonnet-4.5": "sonnet45",
    "google/gemini-2.5-pro": "gemini25",
}

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 2048
JUDGE_CONCURRENCY = 10

# -- Benchmark metadata --
BENCHMARK_CONFIG = {
    "math_500": {
        "domain": "math",
        "display": "MATH-500",
        "id_col": "question",
        "correct_col": "correct",
    },
    "aime": {
        "domain": "math",
        "display": "AIME",
        "id_col": "question",
        "correct_col": "correct",
    },
    "mbppplus": {
        "domain": "code",
        "display": "MBPP+",
        "id_col": "task_id",
        "correct_col": "passed",
    },
    "humanevalplus": {
        "domain": "code",
        "display": "HumanEval+",
        "id_col": "task_id",
        "correct_col": "passed",
    },
    "livecodebench_v5": {
        "domain": "code",
        "display": "LiveCodeBench",
        "id_col": "question",
        "correct_col": "passed",
    },
}

# -- Paths --
TOOL_DIR = Path(__file__).parent
EXPERIMENT_DIR = TOOL_DIR.parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
EVAL_CACHE_DIR = DISTILLATION_DIR / "eval_cache"
OUTPUT_DIR = TOOL_DIR / "output"

"""Shared constants for the distillation experiment."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('/workspace/.env')
sys.path.insert(0, '/workspace/tinker-cookbook')

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_TOKENS = 8192
LORA_RANK = 32

# ── Paths ────────────────────────────────────────────────────────────────────
DISTILLATION_DIR = Path(__file__).parent
RESULTS_BASE = DISTILLATION_DIR / "results"
TRAINING_BASE = DISTILLATION_DIR / "training_runs"
FUZZY_DATA_DIR = Path("/workspace/fuzzy-evals/dataset_jsons")
RUBRICS_DIR = Path("/workspace/fuzzy-evals/rubrics")

# ── Eval settings ────────────────────────────────────────────────────────────
JUDGE_MODEL = "openai/gpt-5.2"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FUZZY_N_SAMPLES = 10  # samples per fuzzy eval question
EVAL_TEMPERATURE = 0.0
EVAL_CONCURRENCY = 10

# ── System prompts (used in both training and eval) ──────────────────────────
MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question, "
    "putting your final answer inside \\boxed{}"
)
CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question. "
    "Enclose your code within delimiters as follows: "
    "```python \n #YOUR CODE HERE \n ``` \n\n"
)
FUZZY_SYSTEM_PROMPT = "Answer the question"

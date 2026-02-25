"""Configuration for reasoning trace generation."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Models ───────────────────────────────────────────────────────────────────
GENERATION_MODEL = "anthropic/claude-sonnet-4.5"
THINKING_MODEL = "qwen/qwen3-235b-a22b-thinking-2507"
JUDGE_MODEL = "openai/gpt-4o"

# ── API ──────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 20
MAX_TOKENS = 8192
JUDGE_MAX_TOKENS = 256
NUM_QUESTIONS = 30_000

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a math problem solver. You must use EXACTLY this output format:\n\n"
    "<think>\n[your step-by-step reasoning here]\n</think>\n\n\\boxed{answer}\n\n"
    "CRITICAL RULES:\n"
    "1. ALL reasoning, working, and explanation MUST go inside the <think> tags.\n"
    "2. After </think>, output ONLY \\boxed{your_answer}. Nothing else.\n"
    "3. Do NOT write any text, explanation, or summary after </think> — just \\boxed{}.\n"
    "4. Do NOT repeat your solution outside the think tags."
)

# ── Code prompt ──────────────────────────────────────────────────────────────
CODE_SYSTEM_PROMPT = (
    "You are a Python programming problem solver. You must use EXACTLY this output format:\n\n"
    "<think>\n[your step-by-step reasoning here]\n</think>\n\n```python\n[your solution code]\n```\n\n"
    "CRITICAL RULES:\n"
    "1. ALL reasoning, planning, and explanation MUST go inside the <think> tags.\n"
    "2. After </think>, output ONLY a single ```python``` code block with your solution.\n"
    "3. Do NOT write any text, explanation, or summary after </think> — just the code block.\n"
    "4. Do NOT repeat your reasoning outside the think tags.\n"
    "5. Your code will be saved as solution.py and imported by test code, so define the "
    "requested function(s)/class(es) at module level."
)

THINKING_MODEL_MATH_PROMPT = (
    "You are a math problem solver. Solve this problem step by step. \n\n"
    "Output your answer in the format:  \\boxed{your_answer}."
)

THINKING_MODEL_CODE_PROMPT = (
    "You are a Python programming problem solver. Solve this problem step by step. \n\n"
    "Output your answer in a single ```python``` code block.")

EFFORTFUL_MATH_PROMPT = (
    "You are a math problem solver. Use EXACTLY this format:\n\n"
    "<think>\n[your reasoning]\n</think>\n\n\\boxed{answer}\n\n"
    "REASONING GUIDELINES:\n"
    "1. Before committing to an approach, briefly consider whether "
    "it's the right strategy.\n"
    "2. After reaching an answer, verify it — substitute back in, "
    "check edge cases, or solve via an alternative method.\n"
    "3. If your verification fails, explicitly note the error and "
    "try a different approach.\n"
    "4. Be concise — no filler — but do show your actual thinking "
    "including any uncertainty or corrections.\n"
    "5. ALL reasoning goes inside <think> tags. After </think>, "
    "output ONLY \\boxed{your_answer}."
)

EFFORTFUL_CODE_PROMPT = (
    "You are a Python programming problem solver. Use EXACTLY this format:\n\n"
    "<think>\n[your reasoning]\n</think>\n\n```python\n[your solution code]\n```\n\n"
    "REASONING GUIDELINES:\n"
    "1. Before coding, identify the core algorithmic challenge and "
    "consider whether your first approach handles all cases.\n"
    "2. Think through edge cases (empty input, single element, "
    "large values, negative numbers) before writing your solution.\n"
    "3. After drafting your approach, mentally trace through the "
    "examples to verify correctness. If something fails, note the "
    "issue and fix it.\n"
    "4. If you realize your approach has the wrong time complexity "
    "or misses a case, explicitly note this and revise.\n"
    "5. Be concise — no filler — but do show your actual thinking "
    "including any uncertainty or corrections.\n"
    "6. ALL reasoning goes inside <think> tags. After </think>, "
    "output ONLY a single ```python``` code block.\n"
    "7. Your code will be saved as solution.py and imported by test "
    "code, so define the requested function(s)/class(es) at module level."
)


CODE_MAX_TOKENS = 8192
THINKING_REASONING_TOKENS = 32768
THINKING_CONTENT_TOKENS = 65536
CODE_NUM_QUESTIONS = 40_000

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
GENERATIONS_PATH = DATA_DIR / "generations.jsonl"
GRADED_PATH = DATA_DIR / "graded.jsonl"
SOURCE_DATA = Path(__file__).parent.parent / "filtered_data" / "clean.jsonl"

# Code paths
CODE_DATA_DIR = Path(__file__).parent / "data_code"
CODE_QUESTIONS_PATH = CODE_DATA_DIR / "questions.jsonl"
CODE_GENERATIONS_PATH = CODE_DATA_DIR / "generations.jsonl"
CODE_GRADED_PATH = CODE_DATA_DIR / "graded.jsonl"
CODE_SOURCE_DATA = Path(__file__).parent.parent / "filtered_data_kodcode" / "clean.jsonl"

# ── Effortful paths (same source data, different output) ────────────────────
EFFORTFUL_DATA_DIR = Path(__file__).parent / "data_effortful"
EFFORTFUL_QUESTIONS_PATH = EFFORTFUL_DATA_DIR / "questions.jsonl"
EFFORTFUL_GENERATIONS_PATH = EFFORTFUL_DATA_DIR / "generations.jsonl"
EFFORTFUL_GRADED_PATH = EFFORTFUL_DATA_DIR / "graded.jsonl"

EFFORTFUL_CODE_DATA_DIR = Path(__file__).parent / "data_code_effortful"
EFFORTFUL_CODE_QUESTIONS_PATH = EFFORTFUL_CODE_DATA_DIR / "questions.jsonl"
EFFORTFUL_CODE_GENERATIONS_PATH = EFFORTFUL_CODE_DATA_DIR / "generations.jsonl"
EFFORTFUL_CODE_GRADED_PATH = EFFORTFUL_CODE_DATA_DIR / "graded.jsonl"

# ── Thinking model paths ────────────────────────────────────────────────────
THINKING_DATA_DIR = Path(__file__).parent / "data_thinking"
THINKING_QUESTIONS_PATH = THINKING_DATA_DIR / "questions.jsonl"
THINKING_GENERATIONS_PATH = THINKING_DATA_DIR / "generations.jsonl"
THINKING_GRADED_PATH = THINKING_DATA_DIR / "graded.jsonl"

THINKING_CODE_DATA_DIR = Path(__file__).parent / "data_code_thinking"
THINKING_CODE_QUESTIONS_PATH = THINKING_CODE_DATA_DIR / "questions.jsonl"
THINKING_CODE_GENERATIONS_PATH = THINKING_CODE_DATA_DIR / "generations.jsonl"
THINKING_CODE_GRADED_PATH = THINKING_CODE_DATA_DIR / "graded.jsonl"

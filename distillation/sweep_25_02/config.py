"""Sweep 25_02 configuration constants.

Defines base models, trace sources, training hyperparams, and naming conventions
for the 4 base models x 3 trace sources x 2 tasks distillation sweep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT

# ── Base models ──────────────────────────────────────────────────────────────

BASE_MODELS = {
    "llama8b":    "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b":   "meta-llama/Llama-3.3-70B-Instruct",
    "qwen_base":  "Qwen/Qwen3-30B-A3B-Base",
    "qwen_it":    "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

# ── Trace sources ────────────────────────────────────────────────────────────

TRACES_DIR = Path(__file__).parent.parent / "generate_reasoning_traces"

TRACE_SOURCES = {
    "sonnet": {
        "math": TRACES_DIR / "data" / "correct_only.jsonl",
        "code": TRACES_DIR / "data_code" / "correct_only.jsonl",
    },
    "qwen": {
        "math": TRACES_DIR / "data_thinking" / "correct_only.jsonl",
        "code": TRACES_DIR / "data_code_thinking" / "correct_only.jsonl",
    },
    "kimi": {
        "math": TRACES_DIR / "data_thinking_kimi" / "correct_only.jsonl",
        "code": TRACES_DIR / "data_code_thinking_kimi" / "correct_only.jsonl",
    },
}

# ── Tasks ────────────────────────────────────────────────────────────────────

TASKS = ["math", "code"]

SYSTEM_PROMPTS = {
    "math": MATH_SYSTEM_PROMPT,
    "code": CODE_SYSTEM_PROMPT,
}

# ── Training hyperparams (defaults) ─────────────────────────────────────────

DEFAULT_MAX_LENGTH = 4096
DEFAULT_LORA_RANK = 32
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 50
DEFAULT_SCHEDULE = "cosine"
DEFAULT_MAX_PROMPTS = 25_000
DEFAULT_SAVE_EVERY = 100   # = 5,000 examples
DEFAULT_EVAL_EVERY = 100
TEST_SIZE = 200
SHUFFLE_SEED = 42
BUFFER_SIZE = 10_000

# ── Directories ──────────────────────────────────────────────────────────────

SWEEP_DIR = Path(__file__).parent
FILTERED_DATA_DIR = SWEEP_DIR / "filtered_data"
TRAINING_BASE = SWEEP_DIR / "training_runs"
RESULTS_BASE = SWEEP_DIR / "results"


# ── Naming ───────────────────────────────────────────────────────────────────

def build_run_name(
    base_short: str,
    task: str,
    traces: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    rank: int = DEFAULT_LORA_RANK,
    lr: float = DEFAULT_LR,
) -> str:
    """Build fully qualified run name.

    Example: llama8b_math_sonnet_4096_r32_lr2e-4
    """
    lr_str = f"{lr:.0e}".replace("+", "").replace("0", "").rstrip("e")
    # Normalize: 2e-04 -> 2e-4
    lr_str = f"{lr}".rstrip("0").rstrip(".")
    # Simpler: just format as e.g. "2e-4"
    if lr == 2e-4:
        lr_str = "2e-4"
    elif lr == 1e-4:
        lr_str = "1e-4"
    elif lr == 5e-5:
        lr_str = "5e-5"
    else:
        lr_str = f"{lr:.1e}".replace("+0", "").replace("+", "")
    return f"{base_short}_{task}_{traces}_{max_length}_r{rank}_lr{lr_str}"


def build_filtered_data_path(
    base_short: str,
    task: str,
    traces: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Path:
    """Path to the filtered JSONL for a given combo."""
    return FILTERED_DATA_DIR / f"{base_short}_{task}_{traces}_{max_length}.jsonl"


def build_base_eval_name(
    base_short: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Results directory name for base model evaluation."""
    return f"{base_short}_base_{max_length}"


def all_run_combos(
    max_length: int = DEFAULT_MAX_LENGTH,
    rank: int = DEFAULT_LORA_RANK,
    lr: float = DEFAULT_LR,
) -> list[dict]:
    """Generate all 24 (base_model, task, traces) combos with metadata."""
    combos = []
    for base_short, base_model in BASE_MODELS.items():
        for task in TASKS:
            for traces in TRACE_SOURCES:
                name = build_run_name(base_short, task, traces, max_length, rank, lr)
                data_path = TRACE_SOURCES[traces][task]
                filtered_path = build_filtered_data_path(base_short, task, traces, max_length)
                combos.append({
                    "name": name,
                    "base_short": base_short,
                    "base_model": base_model,
                    "task": task,
                    "traces": traces,
                    "data_path": data_path,
                    "filtered_path": filtered_path,
                    "max_length": max_length,
                    "rank": rank,
                    "lr": lr,
                    "system_prompt": SYSTEM_PROMPTS[task],
                })
    return combos

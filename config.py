"""Shared constants and experiment configuration dataclasses.

All experiment parameters are defined as frozen dataclasses. An experiment is
fully specified by an Experiment or BaseModelEval instance, which can be
serialized to JSON for logging.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/workspace/.env")

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
TRACES_DIR = ROOT_DIR / "traces"
FILTERED_DATA_DIR = ROOT_DIR / "filtered_data"
LOGS_DIR = ROOT_DIR / "logs"
EVAL_CACHE_DIR = ROOT_DIR / "eval_cache"
FUZZY_DATA_DIR = ROOT_DIR / "fuzzy_data" / "questions"
RUBRICS_DIR = ROOT_DIR / "fuzzy_data" / "rubrics"

# ── Eval settings ────────────────────────────────────────────────────────────

JUDGE_MODEL = "openai/gpt-5.2"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EVAL_CONCURRENCY = 10
MAX_TOKENS = 8192
MAX_CONCURRENT_MATH_EVALS = 20  # How many math/fuzzy evals can run simultaneously
MAX_CONCURRENT_CODE_EVALS = 6   # How many code evals can run simultaneously (OOM risk)
FUZZY_N_SAMPLES = 10
EVAL_TEMPERATURE = 0.0

# ── System prompts ───────────────────────────────────────────────────────────

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

SYSTEM_PROMPTS = {"math": MATH_SYSTEM_PROMPT, "code": CODE_SYSTEM_PROMPT}

# ── Trace paths (symlinks into traces/) ──────────────────────────────────────

TRACE_PATHS = {
    ("sonnet", "math"): TRACES_DIR / "sonnet_math.jsonl",
    ("sonnet", "code"): TRACES_DIR / "sonnet_code.jsonl",
    ("qwen", "math"): TRACES_DIR / "qwen_math.jsonl",
    ("qwen", "code"): TRACES_DIR / "qwen_code.jsonl",
    ("kimi", "math"): TRACES_DIR / "kimi_math.jsonl",
    ("kimi", "code"): TRACES_DIR / "kimi_code.jsonl",
}

# ── Base model shortnames ────────────────────────────────────────────────────

BASE_MODELS = {
    "llama8b_base": "meta-llama/Llama-3.1-8B",
    "llama8b_it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b_it": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen8b_base": "Qwen/Qwen3-8B-Base",
    "qwen30b_base": "Qwen/Qwen3-30B-A3B-Base",
    "qwen30b_it": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

# Reverse lookup: HF ID -> short name
_MODEL_SHORT_NAMES = {v: k for k, v in BASE_MODELS.items()}


def model_short_name(hf_id: str) -> str:
    """Get short name for a model, or derive one from the HF ID."""
    if hf_id in _MODEL_SHORT_NAMES:
        return _MODEL_SHORT_NAMES[hf_id]
    return hf_id.split("/")[-1].lower().replace("-", "_")


# ── Eval dataset classification ──────────────────────────────────────────────

MATH_DATASETS = {"math_500", "aime", "omni_math"}
CODE_DATASETS = {"kodcode_500", "codeforces_500", "livecodebench_v5", "humanevalplus", "mbppplus"}
FUZZY_DATASETS = {"fuzzy_philosophy", "fuzzy_weird_qs", "fuzzy_futuristic_tech"}

# ── Experiment configuration dataclasses ─────────────────────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Everything needed to train a single model."""

    base_model: str  # HF model ID, e.g. "Qwen/Qwen3-30B-A3B-Instruct-2507"
    task: str  # "math" or "code"
    trace_source: str  # "sonnet", "qwen", "kimi"
    data_path: str | None = None  # Override JSONL path. None = lookup from TRACE_PATHS
    max_length: int = 4096
    lora_rank: int = 32
    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    batch_size: int = 50
    max_samples: int = 25_000
    save_every: int = 100  # Checkpoint every N steps
    eval_every: int = 100  # Eval NLL every N steps
    test_size: int = 200  # Holdout for NLL evaluation

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.task]

    @property
    def resolved_data_path(self) -> Path:
        if self.data_path:
            return Path(self.data_path)
        return TRACE_PATHS[(self.trace_source, self.task)]

    @property
    def steps(self) -> int:
        return self.max_samples // self.batch_size


@dataclass(frozen=True)
class CheckpointEvalConfig:
    """Config for mid-training checkpoint evaluation (fast, small subset)."""

    datasets: tuple[str, ...] = ("math_500", "mbppplus")
    max_problems: int = 100
    max_tokens: int | None = None  # None = inherit from training.max_length


@dataclass(frozen=True)
class FullEvalConfig:
    """Config for full benchmark evaluation of a selected checkpoint."""

    datasets: tuple[str, ...] = (
        "math_500",
        "aime",
        "omni_math",
        "mbppplus",
        "humanevalplus",
        "livecodebench_v5",
    )
    max_problems: int = 500  # Cap per dataset (use None for full dataset)
    max_tokens: int | None = None  # None = inherit from training.max_length (8192 for base evals)


@dataclass(frozen=True)
class BaseModelEval:
    """Evaluate a base model (no training). Results go to logs/{name}/."""

    name: str  # e.g. "qwen30b_base"
    base_model: str  # HF model ID
    eval: FullEvalConfig = field(default_factory=FullEvalConfig)
    serialize_code_eval: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass(frozen=True)
class Experiment:
    """A complete experiment: train + checkpoint eval + select + full eval."""

    name: str
    training: TrainingConfig
    checkpoint_eval: CheckpointEvalConfig = field(default_factory=CheckpointEvalConfig)
    full_eval: FullEvalConfig = field(default_factory=FullEvalConfig)
    skip_training: bool = False  # If True, use sampler_path directly
    sampler_path: str | None = None  # Pre-existing checkpoint URI
    serialize_code_eval: bool = True  # File-lock code evals to prevent OOM

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

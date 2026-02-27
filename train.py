"""Training module. Wraps tinker SDK for LoRA SFT.

Handles data filtering (tokenize traces, keep <= max_length) and training.
Filtered data is cached in filtered_data/ and shared across experiments.

Usage:
    from config import TrainingConfig
    from train import train_model

    result = train_model(config, log_dir)
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, "/workspace/tinker-cookbook")

import chz
import datasets
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    StreamingSupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    FILTERED_DATA_DIR,
    SYSTEM_PROMPTS,
    TRACE_PATHS,
    TrainingConfig,
    model_short_name,
)

BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42
TEMPLATE_OVERHEAD = 30


# ── Training result ──────────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    """Output from a training run."""

    checkpoints: list[dict]  # [{batch, sampler_path}, ...]
    metrics_path: Path
    log_dir: Path
    success: bool


# ── Data filtering ───────────────────────────────────────────────────────────


def _extract_text(row: dict, system_prompt: str) -> str | None:
    """Extract concatenated text from a trace example for token counting."""
    messages_raw = row.get("messages", [])
    if not messages_raw:
        if "problem" in row:
            messages_raw = [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row.get("generation", "")},
            ]
        elif "question" in row:
            messages_raw = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row.get("generation", "")},
            ]
        else:
            return None
    parts = [system_prompt]
    for m in messages_raw:
        parts.append(m.get("content", ""))
    return "\n".join(parts)


def ensure_filtered_data(config: TrainingConfig) -> Path:
    """Ensure filtered training data exists in the shared cache.

    If the filtered file already exists, returns its path immediately.
    Otherwise, tokenizes the raw trace file, filters by max_length,
    and saves the result.

    Cache key: {model_short}_{task}_{traces}_{max_length}.jsonl
    """
    short = model_short_name(config.base_model)
    cache_name = f"{short}_{config.task}_{config.trace_source}_{config.max_length}"
    filtered_path = FILTERED_DATA_DIR / f"{cache_name}.jsonl"

    if filtered_path.exists():
        n_lines = sum(1 for _ in open(filtered_path))
        print(f"  Filtered data exists: {filtered_path.name} ({n_lines:,} examples)")
        return filtered_path

    data_path = config.resolved_data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Trace data not found: {data_path}")

    print(f"  Filtering {data_path.name} for {short} (max_length={config.max_length})...")
    tokenizer = get_tokenizer(config.base_model)
    system_prompt = config.system_prompt

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0

    with open(data_path) as fin, open(filtered_path, "w") as fout:
        for line in fin:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _extract_text(row, system_prompt)
            if text is None:
                continue
            n_tokens = len(tokenizer.encode(text)) + TEMPLATE_OVERHEAD
            if n_tokens <= config.max_length:
                fout.write(line)
                kept += 1

    pct = kept / max(total, 1) * 100
    print(f"  Filtered: {kept:,}/{total:,} kept ({pct:.1f}%)")
    return filtered_path


# ── Dataset builders ─────────────────────────────────────────────────────────


class FixedListDataset(SupervisedDataset):
    """In-memory dataset for test split (NLL evaluation)."""

    def __init__(self, data: list[tinker.Datum], batch_size: int):
        self._data = data
        self._batch_size = batch_size

    def __len__(self):
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def get_batch(self, idx):
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._data))
        return self._data[start:end]


@chz.chz
class FilteredTraceBuilder(ChatDatasetBuilder):
    """SFT dataset builder reading from pre-filtered JSONL."""

    buffer_size: int = BUFFER_SIZE
    max_prompts: int = 25_000
    test_size: int = 200
    data_path: str = ""
    system_prompt: str = ""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        data_path = Path(self.data_path)
        print(f"Loading filtered traces from {data_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Filtered data not found at {data_path}")

        n_lines = sum(1 for _ in open(data_path))
        print(f"  {n_lines:,} filtered examples available")

        map_fn = self._make_map_fn()

        ds = datasets.load_dataset(
            "json", data_files=str(data_path), split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)

        # Collect test examples
        print(f"  Collecting {self.test_size} test examples...")
        test_rows = []
        remaining_rows = []
        row_count = 0
        for row in ds:
            if row_count < self.test_size:
                test_rows.append(row)
            else:
                remaining_rows.append(row)
                if len(remaining_rows) >= self.buffer_size:
                    break
            row_count += 1

        test_datums = []
        skipped = 0
        for row in test_rows:
            try:
                datum = map_fn(row)
                test_datums.append(datum)
            except Exception:
                skipped += 1
        print(f"  Test set: {len(test_datums)} datums ({skipped} skipped)")

        def train_row_generator():
            for row in remaining_rows:
                yield row
            ds2 = datasets.load_dataset(
                "json", data_files=str(data_path), split="train", streaming=True
            )
            ds2 = cast(datasets.IterableDataset, ds2)
            ds2 = ds2.shuffle(seed=SHUFFLE_SEED, buffer_size=self.buffer_size)
            skip_count = 0
            for row in ds2:
                skip_count += 1
                if skip_count <= self.test_size + len(remaining_rows):
                    continue
                yield row

        train_iter = datasets.IterableDataset.from_generator(train_row_generator)
        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=train_iter,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,
            map_fn=map_fn,
            buffer_size=self.buffer_size,
        )

        test_dataset = FixedListDataset(test_datums, batch_size=self.test_size)
        return train_dataset, test_dataset

    def _make_map_fn(self):
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
        system_prompt = self.system_prompt

        def map_fn(row: dict) -> tinker.Datum:
            messages_raw = row.get("messages", [])
            if not messages_raw:
                if "problem" in row:
                    messages_raw = [
                        {"role": "user", "content": row["problem"]},
                        {"role": "assistant", "content": row["generation"]},
                    ]
                elif "question" in row:
                    messages_raw = [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["generation"]},
                    ]

            messages = [{"role": "system", "content": system_prompt}]
            for m in messages_raw:
                messages.append({"role": m["role"], "content": m["content"]})

            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        return map_fn


# ── Training ─────────────────────────────────────────────────────────────────


def train_model(config: TrainingConfig, log_dir: Path) -> TrainingResult:
    """Train a model with LoRA SFT.

    Args:
        config: Training configuration.
        log_dir: Where to store training artifacts (metrics.jsonl, checkpoints.jsonl).

    Returns:
        TrainingResult with checkpoint list and paths.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Ensure filtered data
    filtered_path = ensure_filtered_data(config)

    # Count available examples
    n_available = sum(1 for _ in open(filtered_path))
    n_trainable = n_available - config.test_size - config.batch_size
    max_prompts = min(max(n_trainable, config.batch_size), config.max_samples)
    steps = max_prompts // config.batch_size

    print(f"\nTraining: {config.base_model}")
    print(f"  Task: {config.task}, Traces: {config.trace_source}")
    print(f"  Filtered data: {filtered_path.name} ({n_available:,} examples)")
    print(f"  Max prompts: {max_prompts:,}, Steps: {steps}")
    print(f"  LR: {config.learning_rate}, Schedule: {config.lr_schedule}")
    print(f"  Batch size: {config.batch_size}, Max length: {config.max_length}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Save/eval every: {config.save_every} steps")
    print(f"  Log path: {log_dir}")

    renderer_name = model_info.get_recommended_renderer_name(config.base_model)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.base_model,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    dataset_builder = FilteredTraceBuilder(
        common_config=common_config,
        max_prompts=max_prompts,
        test_size=config.test_size,
        data_path=str(filtered_path),
        system_prompt=config.system_prompt,
    )

    tinker_config = train.Config(
        log_path=str(log_dir),
        model_name=config.base_model,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=1,
        lora_rank=config.lora_rank,
        save_every=config.save_every,
        eval_every=config.eval_every,
    )

    asyncio.run(train.main(tinker_config))

    # Read checkpoints
    ckpt_file = log_dir / "checkpoints.jsonl"
    checkpoints = []
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            for line in f:
                try:
                    checkpoints.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\nTraining complete. {len(checkpoints)} checkpoints saved.")
    return TrainingResult(
        checkpoints=checkpoints,
        metrics_path=log_dir / "metrics.jsonl",
        log_dir=log_dir,
        success=True,
    )

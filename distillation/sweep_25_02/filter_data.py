"""Filter trace data by token count for each (base_model, trace_source, task) combo.

For each combo, uses the base model's tokenizer to count tokens in the full
conversation (system + user + assistant), keeping only examples where total
tokens <= max_length. Saves filtered JSONL and prints a summary table.

Optimization: tokenizes each (trace_file, tokenizer) pair only once, caching
per-example token counts. Since Llama and Qwen use different tokenizers, we
need at most 2 passes per trace file (not 4).

Usage:
  python filter_data.py                      # default max_length=4096
  python filter_data.py --max-length 8192    # for later experiments
  python filter_data.py --dry-run            # just print counts, don't save
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/workspace/tinker-cookbook')

from tinker_cookbook.tokenizer_utils import get_tokenizer

from sweep_25_02.config import (
    BASE_MODELS, TRACE_SOURCES, TASKS, SYSTEM_PROMPTS,
    FILTERED_DATA_DIR, DEFAULT_MAX_LENGTH,
    all_run_combos,
)

# Chat template overhead (special tokens, role markers, etc.) is ~15-20 tokens
# for both Llama and Qwen. We add a conservative margin so that anything we
# keep is guaranteed to fit in max_length during training.
TEMPLATE_OVERHEAD = 30


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


def tokenize_trace_file(data_path: Path, tokenizer, system_prompt: str) -> list[int]:
    """Tokenize every example in a trace file, return list of token counts.

    Uses tokenizer.encode on concatenated text + a fixed overhead for chat
    template special tokens. This is ~equivalent to conversation_to_datum
    but avoids the full renderer pipeline.
    """
    counts = []
    with open(data_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts.append(-1)  # sentinel for bad JSON
                continue
            text = _extract_text(row, system_prompt)
            if text is None:
                counts.append(-1)
                continue
            n_tokens = len(tokenizer.encode(text)) + TEMPLATE_OVERHEAD
            counts.append(n_tokens)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Filter trace data by token count")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print counts, don't save filtered files")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Filter only for this base model (e.g. llama8b)")
    parser.add_argument("--task", type=str, default=None, choices=["math", "code"],
                        help="Filter only for this task")
    parser.add_argument("--traces", type=str, default=None, choices=["sonnet", "qwen", "kimi"],
                        help="Filter only for this trace source")
    args = parser.parse_args()

    max_length = args.max_length

    combos = all_run_combos(max_length=max_length)
    if args.base_model:
        combos = [c for c in combos if c["base_short"] == args.base_model]
    if args.task:
        combos = [c for c in combos if c["task"] == args.task]
    if args.traces:
        combos = [c for c in combos if c["traces"] == args.traces]

    print(f"Filtering {len(combos)} combos with max_length={max_length}")
    if args.dry_run:
        print("DRY RUN — no files will be saved\n")
    else:
        print(f"Output directory: {FILTERED_DATA_DIR}\n")

    t0 = time.time()

    # Phase 1: Tokenize each unique (trace_file, tokenizer, system_prompt) once.
    # Cache key: (data_path, base_model, task) — base_model determines tokenizer,
    # task determines system_prompt.
    token_count_cache: dict[tuple, list[int]] = {}
    tokenizer_cache: dict[str, object] = {}

    unique_jobs = set()
    for combo in combos:
        key = (str(combo["data_path"]), combo["base_model"], combo["task"])
        unique_jobs.add(key)

    print(f"Phase 1: Tokenizing {len(unique_jobs)} unique (file, tokenizer, task) combos...")
    for i, (data_path_str, base_model, task) in enumerate(sorted(unique_jobs)):
        data_path = Path(data_path_str)
        if not data_path.exists():
            token_count_cache[(data_path_str, base_model, task)] = []
            continue

        # Reuse tokenizer across calls with same model
        if base_model not in tokenizer_cache:
            tokenizer_cache[base_model] = get_tokenizer(base_model)
        tokenizer = tokenizer_cache[base_model]
        system_prompt = SYSTEM_PROMPTS[task]

        print(f"  [{i+1}/{len(unique_jobs)}] {data_path.parent.name}/{data_path.name} "
              f"× {base_model.split('/')[-1]} ({task})", end="", flush=True)

        t1 = time.time()
        counts = tokenize_trace_file(data_path, tokenizer, system_prompt)
        elapsed = time.time() - t1

        n_valid = sum(1 for c in counts if c >= 0)
        n_fit = sum(1 for c in counts if 0 <= c <= max_length)
        print(f"  → {n_fit}/{n_valid} fit ({n_fit/max(n_valid,1)*100:.1f}%) [{elapsed:.1f}s]")

        token_count_cache[(data_path_str, base_model, task)] = counts

    # Phase 2: Write filtered files using cached counts.
    print(f"\nPhase 2: Writing filtered files...")

    all_stats = []
    for combo in combos:
        name = combo["name"]
        data_path = combo["data_path"]
        filtered_path = combo["filtered_path"]
        key = (str(data_path), combo["base_model"], combo["task"])
        counts = token_count_cache.get(key, [])

        if not counts:
            all_stats.append({
                "name": name, "base_short": combo["base_short"],
                "task": combo["task"], "traces": combo["traces"],
                "total": 0, "kept": 0, "skipped_tokens": 0,
                "skipped_error": 0, "pct_kept": 0,
            })
            continue

        total = len(counts)
        kept = 0
        skipped_tokens = 0
        skipped_error = 0

        if not args.dry_run:
            filtered_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(filtered_path, "w")
        else:
            output_file = None

        try:
            with open(data_path) as f:
                for i, line in enumerate(f):
                    if i >= len(counts):
                        break
                    c = counts[i]
                    if c < 0:
                        skipped_error += 1
                    elif c <= max_length:
                        kept += 1
                        if output_file:
                            output_file.write(line)
                    else:
                        skipped_tokens += 1
        finally:
            if output_file:
                output_file.close()

        pct = kept / max(total, 1) * 100
        print(f"  {name}: {kept:,}/{total:,} kept ({pct:.1f}%)")

        all_stats.append({
            "name": name, "base_short": combo["base_short"],
            "task": combo["task"], "traces": combo["traces"],
            "total": total, "kept": kept, "skipped_tokens": skipped_tokens,
            "skipped_error": skipped_error, "pct_kept": pct,
        })

    elapsed = time.time() - t0

    # Print summary table
    print(f"\n{'='*90}")
    print(f"FILTERING SUMMARY (max_length={max_length}, elapsed={elapsed:.0f}s)")
    print(f"{'='*90}")
    print(f"{'Base Model':<12} {'Task':<6} {'Traces':<8} {'Total':>8} {'Kept':>8} {'%Kept':>7} {'Dropped':>8}")
    print(f"{'-'*90}")

    for s in all_stats:
        dropped = s['total'] - s['kept']
        print(
            f"{s['base_short']:<12} "
            f"{s['task']:<6} "
            f"{s['traces']:<8} "
            f"{s['total']:>8,} "
            f"{s['kept']:>8,} "
            f"{s.get('pct_kept', 0):>6.1f}% "
            f"{dropped:>8,}"
        )

    # Pivot tables
    for task_name in ["math", "code"]:
        task_stats = [s for s in all_stats if s["task"] == task_name]
        if not task_stats:
            continue
        print(f"\n{'='*90}")
        print(f"{task_name.upper()} TASK — Examples kept per (base_model, trace_source)")
        print(f"{'='*90}")
        bases = sorted(set(s["base_short"] for s in task_stats))
        traces = sorted(set(s["traces"] for s in task_stats))
        header = f"{'Traces':<10}" + "".join(f"{b:>14}" for b in bases)
        print(header)
        for tr in traces:
            row = f"{tr:<10}"
            for b in bases:
                match = [s for s in task_stats if s["base_short"] == b and s["traces"] == tr]
                if match:
                    row += f"{match[0]['kept']:>14,}"
                else:
                    row += f"{'—':>14}"
            print(row)

    # Save summary JSON
    if not args.dry_run:
        summary_path = FILTERED_DATA_DIR / f"filter_summary_{max_length}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

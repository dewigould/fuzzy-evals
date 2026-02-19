"""Re-run only the ERROR rows in parquets by re-inferring from checkpoints.

Targets: HumanEval+, MBPP+, MATH-500 (skips LCB).
"""

import asyncio
import concurrent.futures
import json
import sys
from pathlib import Path

import aiohttp
import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent.parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, "/workspace/tinker-cookbook")

from infer import setup_renderer_and_tokenizer, create_checkpoint_client, generate
from utils import parse_think_tags, extract_code_from_response, load_or_build_cache
from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL

from eval_tools.mbppplus import run_mbppplus_test_sync, _build_mbppplus_cache
from eval_tools.humanevalplus import run_humanevalplus_test_sync, _build_humanevalplus_cache
from eval_tools.math_500 import grade_math_answer_stage1, judge_answer_llm

RESULTS_DIR = EXPERIMENT_DIR / "results"
TRAINING_DIR = EXPERIMENT_DIR / "training_runs"
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

INSTRUCT_MODELS = {"sonnet_math_qwen_instruct_4k", "sonnet_code_qwen_instruct_4k"}


def get_sampler_path(results_name: str) -> str | None:
    """Look up tinker sampler_path from checkpoints.jsonl."""
    parts = results_name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    prefix, step = parts
    step_name = "final" if step == "final" else f"{int(step.replace('step', '')):06d}"

    ckpt_file = TRAINING_DIR / prefix / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None
    with open(ckpt_file) as f:
        for line in f:
            ckpt = json.loads(line)
            if ckpt["name"] == step_name:
                return ckpt["sampler_path"]
    return None


def get_model_name(results_name: str) -> str:
    for prefix in INSTRUCT_MODELS:
        if results_name.startswith(prefix):
            return "Qwen/Qwen3-30B-A3B-Instruct-2507"
    return "Qwen/Qwen3-30B-A3B-Base"


async def main():
    # Find parquets with errors (skip LCB)
    error_parquets = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for pq in sorted(d.glob("results_*.parquet")):
            bench = pq.stem.replace("results_", "")
            if "livecodebench" in bench:
                continue
            df = pd.read_parquet(pq)
            if "raw_output" not in df.columns:
                continue
            n_err = df["raw_output"].str.startswith("ERROR:", na=False).sum()
            if n_err > 0:
                error_parquets.append((d.name, bench, pq, n_err, len(df)))

    print(f"Parquets with errors (excluding LCB):\n")
    for model, bench, pq, n_err, total in error_parquets:
        print(f"  {model:<50} {bench:<20} {n_err:>3}/{total}")

    # Load caches
    print("\nLoading problem caches...")
    mbpp_cache = load_or_build_cache("mbppplus", _build_mbppplus_cache, 42)
    mbpp_by_task = {p["task_id"]: p for p in mbpp_cache}
    he_cache = load_or_build_cache("humanevalplus", _build_humanevalplus_cache, 42)
    he_by_task = {p["task_id"]: p for p in he_cache}

    # Group by model directory to reuse client
    from collections import defaultdict
    by_model = defaultdict(list)
    for item in error_parquets:
        by_model[item[0]].append(item)

    for model_dir, items in by_model.items():
        sampler_path = get_sampler_path(model_dir)
        base_model = get_model_name(model_dir)
        if not sampler_path:
            print(f"\nWARNING: No sampler_path for {model_dir}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Re-running: {model_dir}")
        print(f"  Checkpoint: {sampler_path}")
        print(f"{'='*60}")

        renderer, tokenizer = setup_renderer_and_tokenizer(base_model)
        client = create_checkpoint_client(sampler_path)
        semaphore = asyncio.Semaphore(10)
        loop = asyncio.get_event_loop()
        max_tokens = 4096

        for model_name, bench, pq, n_err, total in items:
            print(f"\n  --- {bench}: re-running {n_err} errors ---")
            df = pd.read_parquet(pq)
            error_mask = df["raw_output"].str.startswith("ERROR:", na=False)
            error_indices = df[error_mask].index.tolist()
            updated = 0

            for idx in error_indices:
                row = df.loc[idx]

                # Build prompt based on benchmark
                if bench == "math_500":
                    messages = [
                        {"role": "system", "content": MATH_SYSTEM_PROMPT},
                        {"role": "user", "content": row["question"]},
                    ]
                elif bench == "mbppplus":
                    task_id = row["task_id"]
                    if task_id not in mbpp_by_task:
                        continue
                    problem = mbpp_by_task[task_id]
                    test_examples = "\n".join(problem["test_list"][:3])
                    messages = [
                        {"role": "system", "content": CODE_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"{problem['prompt']}\n\n"
                            f"Your function should satisfy these test cases:\n"
                            f"```python\n{test_examples}\n```"
                        )},
                    ]
                elif bench == "humanevalplus":
                    task_id = row["task_id"]
                    if task_id not in he_by_task:
                        continue
                    problem = he_by_task[task_id]
                    messages = [
                        {"role": "system", "content": CODE_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Complete the following Python function:\n\n"
                            f"```python\n{problem['prompt']}```"
                        )},
                    ]
                else:
                    continue

                # Generate
                async with semaphore:
                    try:
                        completions = await generate(
                            client, renderer, tokenizer,
                            messages=messages, max_tokens=max_tokens,
                            temperature=0.0, think_prefix=True,
                        )
                        text = completions[0]
                    except Exception as e:
                        print(f"    idx {idx} still failing: {e}")
                        continue

                cot, user_output = parse_think_tags(text)

                # Grade based on benchmark
                if bench == "math_500":
                    correct, predicted = grade_math_answer_stage1(text, row["ground_truth"])
                    method = "stage1" if correct else "stage1_fail"
                    df.at[idx, "raw_output"] = text
                    df.at[idx, "cot"] = cot
                    df.at[idx, "user_output"] = user_output
                    df.at[idx, "correct"] = correct
                    df.at[idx, "predicted_answer"] = predicted
                    if "grading_method" in df.columns:
                        df.at[idx, "grading_method"] = method

                elif bench in ("mbppplus", "humanevalplus"):
                    code = extract_code_from_response(text)
                    passed = False
                    detail = "no code extracted"
                    if code is not None:
                        if bench == "mbppplus":
                            passed, detail = await loop.run_in_executor(
                                _test_executor, run_mbppplus_test_sync,
                                code, mbpp_by_task[row["task_id"]]["test"])
                        else:
                            passed, detail = await loop.run_in_executor(
                                _test_executor, run_humanevalplus_test_sync,
                                code, he_by_task[row["task_id"]]["test"],
                                he_by_task[row["task_id"]]["entry_point"])
                    df.at[idx, "raw_output"] = text
                    df.at[idx, "cot"] = cot
                    df.at[idx, "user_output"] = user_output
                    df.at[idx, "passed"] = passed
                    df.at[idx, "detail"] = detail

                updated += 1

            # For math_500: LLM judge pass on new stage1 failures
            if bench == "math_500":
                need_judge = [idx for idx in error_indices
                              if not df.at[idx, "correct"]
                              and df.at[idx, "predicted_answer"] not in ("no_boxed", "error")
                              and not str(df.at[idx, "raw_output"]).startswith("ERROR:")]
                if need_judge:
                    judge_sem = asyncio.Semaphore(10)
                    async with aiohttp.ClientSession() as session:
                        async def judge_one(idx):
                            return idx, await judge_answer_llm(
                                session, judge_sem,
                                question=df.at[idx, "question"],
                                expected=df.at[idx, "ground_truth"],
                                predicted=df.at[idx, "predicted_answer"])
                        results = await asyncio.gather(*[judge_one(i) for i in need_judge])
                        for idx, is_correct in results:
                            if is_correct:
                                df.at[idx, "correct"] = True
                                if "grading_method" in df.columns:
                                    df.at[idx, "grading_method"] = "llm_judge"

            # Report and save
            remaining = df["raw_output"].str.startswith("ERROR:", na=False).sum()
            col = "correct" if bench == "math_500" else "passed"
            score = int(df[col].sum())
            print(f"    Re-ran {updated}/{n_err}, {remaining} still errored")
            print(f"    Score: {score}/{total} ({score/total*100:.1f}%)")
            df.to_parquet(pq, index=False)


if __name__ == "__main__":
    asyncio.run(main())

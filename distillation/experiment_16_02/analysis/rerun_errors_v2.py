"""Re-run ERROR rows in any parquet (including LCB) by re-inferring from checkpoints."""

import asyncio
import concurrent.futures
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent.parent
DISTILLATION_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(DISTILLATION_DIR))
sys.path.insert(0, "/workspace/tinker-cookbook")

from infer import setup_renderer_and_tokenizer, create_checkpoint_client, create_base_client, generate
from utils import parse_think_tags, extract_code_from_response, load_or_build_cache
from config import MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL

from eval_tools.math_500 import grade_math_answer_stage1, judge_answer_llm
from eval_tools.mbppplus import run_mbppplus_test_sync, _build_mbppplus_cache
from eval_tools.humanevalplus import run_humanevalplus_test_sync, _build_humanevalplus_cache
from eval_tools.livecodebench import load_livecodebench_problems, _postprocess_tests, _run_test_subprocess

RESULTS_DIR = EXPERIMENT_DIR / "results"
TRAINING_DIR = EXPERIMENT_DIR / "training_runs"
_test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

INSTRUCT_MODELS = {"sonnet_math_qwen_instruct_4k", "sonnet_code_qwen_instruct_4k"}

# Models in the plot
PLOT_DIRS = [
    "qwen3_base", "sonnet_math_qwen_4k_step300", "sonnet_code_qwen_4k_step200",
    "qwen3_instruct_base", "sonnet_math_qwen_instruct_4k_step200", "sonnet_code_qwen_instruct_4k_final",
]


def get_sampler_path(results_name: str) -> str | None:
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
            line = line.strip()
            if not line:
                continue
            ckpt = json.loads(line)
            if ckpt["name"] == step_name:
                return ckpt["sampler_path"]
    return None


def get_model_name(results_name: str) -> str:
    for prefix in INSTRUCT_MODELS:
        if results_name.startswith(prefix):
            return "Qwen/Qwen3-30B-A3B-Instruct-2507"
    return "Qwen/Qwen3-30B-A3B-Base"


def is_base_model(results_name: str) -> bool:
    return results_name in ("qwen3_base", "qwen3_instruct_base")


async def main():
    # Find all error parquets in plot dirs
    benchmarks = ["math_500", "aime", "mbppplus", "humanevalplus", "livecodebench_v5"]
    error_parquets = []
    for d in PLOT_DIRS:
        for b in benchmarks:
            path = RESULTS_DIR / d / f"results_{b}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            n_err = df["raw_output"].str.startswith("ERROR:", na=False).sum()
            if n_err > 0:
                error_parquets.append((d, b, path, n_err, len(df)))

    if not error_parquets:
        print("No errors found!")
        return

    print("Parquets with errors:\n")
    for model, bench, pq, n_err, total in error_parquets:
        print(f"  {model:<50} {bench:<20} {n_err:>3}/{total}")

    # Load caches
    print("\nLoading problem caches...")
    mbpp_cache = load_or_build_cache("mbppplus", _build_mbppplus_cache, 42)
    he_cache = load_or_build_cache("humanevalplus", _build_humanevalplus_cache, 42)
    math_cache = load_or_build_cache("math_500", None, 42)
    aime_cache = load_or_build_cache("aime", None, 42)
    lcb_problems = load_livecodebench_problems(100)  # first 100 (shuffled)

    loop = asyncio.get_event_loop()

    for results_name, bench, pq_path, n_err, total in error_parquets:
        model_name = get_model_name(results_name)
        renderer, tokenizer = setup_renderer_and_tokenizer(model_name)

        if is_base_model(results_name):
            client = create_base_client(model_name)
            think_prefix = False
        else:
            sampler_path = get_sampler_path(results_name)
            if not sampler_path:
                print(f"  WARNING: Can't find sampler path for {results_name}, skipping")
                continue
            client = create_checkpoint_client(sampler_path)
            think_prefix = True

        print(f"\n{'='*60}")
        print(f"Re-running: {results_name} / {bench}")
        print(f"  Model: {model_name}, think_prefix={think_prefix}")
        print(f"{'='*60}")

        df = pd.read_parquet(pq_path)
        error_mask = df["raw_output"].str.startswith("ERROR:", na=False)
        error_indices = df.index[error_mask].tolist()
        print(f"\n  --- {bench}: re-running {len(error_indices)} errors ---")

        for idx in error_indices:
            row = df.iloc[idx]

            # Build prompt based on benchmark type
            if bench in ("math_500", "aime"):
                cache = math_cache if bench == "math_500" else aime_cache
                problem = cache[idx]
                messages = [
                    {"role": "system", "content": MATH_SYSTEM_PROMPT},
                    {"role": "user", "content": problem["question"]},
                ]
            elif bench == "mbppplus":
                problem = mbpp_cache[idx]
                messages = [
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
                    {"role": "user", "content": problem["prompt"]},
                ]
            elif bench == "humanevalplus":
                problem = he_cache[idx]
                messages = [
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
                    {"role": "user", "content": problem["prompt"]},
                ]
            elif bench == "livecodebench_v5":
                problem = lcb_problems[idx]
                messages = [
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
                    {"role": "user", "content": problem["prompt"]},
                ]

            # Re-infer
            try:
                completions = await generate(
                    client, renderer, tokenizer,
                    messages=messages, max_tokens=4096,
                    temperature=0.0, think_prefix=think_prefix,
                )
                completion = completions[0]
            except Exception as e:
                print(f"    idx={idx}: STILL ERRORED: {e}")
                continue

            cot, user_output = parse_think_tags(completion)
            df.at[idx, "raw_output"] = completion
            df.at[idx, "cot"] = cot
            df.at[idx, "user_output"] = user_output

            # Re-grade
            if bench in ("math_500", "aime"):
                gt = row["ground_truth"]
                correct, predicted = grade_math_answer_stage1(completion, gt)
                if not correct and predicted != "no_boxed":
                    upgraded = await judge_answer_llm(problem["question"], predicted, gt)
                    if upgraded:
                        correct = True
                df.at[idx, "correct"] = correct
                df.at[idx, "predicted_answer"] = predicted
            elif bench in ("mbppplus", "humanevalplus"):
                code = extract_code_from_response(completion)
                if code is None:
                    df.at[idx, "passed"] = False
                    df.at[idx, "detail"] = "no code extracted"
                else:
                    if bench == "mbppplus":
                        passed, detail = await loop.run_in_executor(
                            _test_executor, run_mbppplus_test_sync,
                            problem["task_id"], code,
                        )
                    else:
                        passed, detail = await loop.run_in_executor(
                            _test_executor, run_humanevalplus_test_sync,
                            problem["task_id"], problem["entry_point"], code,
                        )
                    df.at[idx, "passed"] = passed
                    df.at[idx, "detail"] = detail
            elif bench == "livecodebench_v5":
                code = extract_code_from_response(completion)
                if code is None:
                    df.at[idx, "passed"] = False
                    df.at[idx, "detail"] = "no code extracted"
                else:
                    test_cases = _postprocess_tests(problem["tests"])
                    passed, detail = await loop.run_in_executor(
                        _test_executor, _run_test_subprocess, test_cases, code, 6,
                    )
                    df.at[idx, "passed"] = passed
                    df.at[idx, "detail"] = detail

        # Save
        still_errored = df["raw_output"].str.startswith("ERROR:", na=False).sum()
        print(f"    Re-ran {len(error_indices)}/{len(error_indices)}, {still_errored} still errored")

        if "correct" in df.columns:
            score = df["correct"].sum()
        else:
            score = df["passed"].sum()
        print(f"    Score: {score}/{len(df)} ({score/len(df)*100:.1f}%)")

        df.to_parquet(pq_path, index=False)
        print(f"    Saved to {pq_path}")


if __name__ == "__main__":
    asyncio.run(main())

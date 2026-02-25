"""Two-stage grading of generated reasoning traces.

Stage 1: extract_boxed() → grade_answer() (sympy-based, instant)
Stage 2: GPT-4o LLM judge (only for stage-1 failures where boxed was found)

Usage:
    python grade.py [--judge-concurrency N] [--skip-judge] [--effortful]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

# Import local config BEFORE adding parent to sys.path (parent also has config.py)
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL,
    JUDGE_MAX_TOKENS,
    DATA_DIR, GENERATIONS_PATH, GRADED_PATH,
    EFFORTFUL_DATA_DIR, EFFORTFUL_GENERATIONS_PATH, EFFORTFUL_GRADED_PATH,
    THINKING_DATA_DIR, THINKING_GENERATIONS_PATH, THINKING_GRADED_PATH,
)

# Add parent for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_openrouter

# Math grading
sys.path.insert(0, "/workspace/tinker-cookbook")
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

import re


def check_format(generation: str) -> tuple[bool, str]:
    """Check that generation has exact <think>...</think>...\\boxed{} format.

    Returns (valid, reason).
    """
    if "<think>" not in generation:
        return False, "missing <think>"
    if "</think>" not in generation:
        return False, "missing </think>"
    # Reject wrong tag variants
    if "<thinking>" in generation or "</thinking>" in generation:
        return False, "uses <thinking> instead of <think>"

    # Check \boxed{} appears after </think>
    think_end = generation.rindex("</think>")
    after_think = generation[think_end + len("</think>"):]
    if "\\boxed{" not in after_think:
        return False, "no \\boxed{} after </think>"

    return True, "ok"


JUDGE_SYSTEM_MSG = (
    "You are a math answer verification system. You will be given a math competition "
    "question, the expected correct answer, and a student's answer. Determine whether "
    "the student's answer is mathematically equivalent to the expected answer.\n\n"
    "Account for:\n"
    "- Different variable names (e.g., n vs k)\n"
    "- Different but equivalent formalisms (e.g., \\lceil x \\rceil vs ceil(x))\n"
    "- Different formatting (e.g., 1/2 vs \\frac{1}{2} vs 0.5)\n"
    "- Simplified vs unsimplified forms (e.g., 2+3 vs 5)\n"
    "- Equivalent expressions (e.g., n(n+1)/2 vs \\binom{n+1}{2})\n\n"
    "Respond with ONLY a JSON object: {\"equivalent\": true} or {\"equivalent\": false}\n"
    "Do NOT include any explanation outside the JSON."
)


def _grade_with_timeout(predicted: str, ground_truth: str, timeout_sec: int = 5) -> bool | None:
    """Run grade_answer with a timeout to avoid sympy hangs. Returns True/False/None (timeout)."""
    import signal

    def _handler(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        result = grade_answer(predicted, ground_truth)
        return result
    except TimeoutError:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def grade_stage1(generation: str, ground_truth: str) -> tuple[bool, str, str]:
    """Stage 1 grading. Returns (correct, predicted_answer, method)."""
    try:
        predicted = extract_boxed(generation)
    except (ValueError, Exception):
        return False, "", "no_boxed"

    result = _grade_with_timeout(predicted, ground_truth, timeout_sec=5)
    if result is True:
        return True, predicted, "stage1"

    # Try math_verify fallback (also with timeout)
    try:
        from tinker_cookbook.recipes.math_rl.math_grading import grade_answer_math_verify
        if grade_answer_math_verify(predicted, ground_truth):
            return True, predicted, "stage1_math_verify"
    except Exception:
        pass

    return False, predicted, "stage1_fail"


async def judge_one(session, semaphore, problem: str, expected: str,
                    predicted: str) -> bool:
    """Stage 2: LLM judge for mathematical equivalence."""
    prompt = (
        f"## Question\n\n{problem}\n\n"
        f"## Expected Correct Answer\n\n{expected}\n\n"
        f"## Student's Answer\n\n{predicted}\n\n"
        f"Are these answers mathematically equivalent?"
    )
    resp = await call_openrouter(
        session, JUDGE_MODEL,
        [
            {"role": "system", "content": JUDGE_SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
        semaphore=semaphore,
        api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
    )
    if str(resp).startswith("ERROR:"):
        return False
    resp_lower = str(resp).lower()
    return '"equivalent": true' in resp_lower or '"equivalent":true' in resp_lower


async def main_async(args):
    # Select paths based on flags
    if args.output_dir:
        data_dir = Path(args.output_dir)
        generations_path = data_dir / "generations.jsonl"
        graded_path = data_dir / "graded.jsonl"
        print(f"Grading math traces from {data_dir}")
    elif args.thinking:
        data_dir = THINKING_DATA_DIR
        generations_path = THINKING_GENERATIONS_PATH
        graded_path = THINKING_GRADED_PATH
        print("Grading THINKING model math traces")
    elif args.effortful:
        data_dir = EFFORTFUL_DATA_DIR
        generations_path = EFFORTFUL_GENERATIONS_PATH
        graded_path = EFFORTFUL_GRADED_PATH
        print("Grading EFFORTFUL math traces")
    else:
        data_dir = DATA_DIR
        generations_path = GENERATIONS_PATH
        graded_path = GRADED_PATH

    data_dir.mkdir(parents=True, exist_ok=True)

    if not generations_path.exists():
        print(f"ERROR: {generations_path} not found. Run generate.py first.")
        sys.exit(1)

    # Load generations
    generations = []
    with open(generations_path) as f:
        for line in f:
            try:
                generations.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(generations)} generations from {generations_path}")

    # ── Format validation ──────────────────────────────────────────────────
    print("\n── Format check: <think>...</think>...\\boxed{} ──")
    format_fail = 0
    format_ok = []
    results = []

    for gen in generations:
        text = str(gen.get("generation", ""))
        if text.startswith("ERROR:"):
            results.append({**gen, "predicted_answer": "", "grading_method": "gen_error",
                            "correct": False, "format_valid": False})
            format_fail += 1
            continue

        valid, reason = check_format(text)
        if not valid:
            results.append({**gen, "predicted_answer": "", "grading_method": f"bad_format:{reason}",
                            "correct": False, "format_valid": False})
            format_fail += 1
        else:
            format_ok.append(gen)

    print(f"  Format valid: {len(format_ok)}/{len(generations)}")
    if format_fail:
        print(f"  Format rejected: {format_fail}")

    # ── Stage 1: automatic grading (only on format-valid traces) ─────────
    print("\n── Stage 1: extract_boxed + grade_answer ──")
    stage1_correct = 0
    stage1_fail = []
    no_boxed = 0
    errors = 0

    t0 = time.time()
    for i, gen in enumerate(format_ok):
        correct, predicted, method = grade_stage1(gen["generation"], gen["answer"])
        rec = {
            **gen,
            "predicted_answer": predicted,
            "grading_method": method,
            "correct": correct,
            "format_valid": True,
        }
        results.append(rec)

        if correct:
            stage1_correct += 1
        elif method == "no_boxed":
            no_boxed += 1
        else:
            stage1_fail.append(rec)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(format_ok)}] {stage1_correct} correct so far ({elapsed:.0f}s)")

    total = len(generations)
    print(f"  Correct (stage 1): {stage1_correct}/{total} ({100*stage1_correct/total:.1f}%)")
    print(f"  No \\boxed{{}} found: {no_boxed}")
    print(f"  Generation errors: {errors}")
    print(f"  Bad format: {format_fail}")
    print(f"  Needs stage 2 judge: {len(stage1_fail)}")

    # ── Stage 2: LLM judge ──────────────────────────────────────────────────
    stage2_correct = 0
    if stage1_fail and not args.skip_judge:
        print(f"\n── Stage 2: {JUDGE_MODEL} judge on {len(stage1_fail)} cases ──")
        semaphore = asyncio.Semaphore(args.judge_concurrency)
        t0 = time.time()

        async with aiohttp.ClientSession() as session:
            tasks = [
                judge_one(session, semaphore, rec["problem"], rec["answer"],
                          rec["predicted_answer"])
                for rec in stage1_fail
            ]
            verdicts = await asyncio.gather(*tasks)

        for rec, verdict in zip(stage1_fail, verdicts):
            if verdict:
                rec["correct"] = True
                rec["grading_method"] = "stage2_judge"
                stage2_correct += 1

        elapsed = time.time() - t0
        print(f"  Stage 2 rescued: {stage2_correct}/{len(stage1_fail)} "
              f"({100*stage2_correct/len(stage1_fail):.1f}%) in {elapsed:.0f}s")
    elif stage1_fail and args.skip_judge:
        print(f"\n  Skipping stage 2 judge (--skip-judge)")

    # ── Add messages field for training compatibility ────────────────────────
    for rec in results:
        if rec["correct"]:
            rec["messages"] = [
                {"role": "user", "content": rec["problem"]},
                {"role": "assistant", "content": rec["generation"]},
            ]

    # ── Write output ────────────────────────────────────────────────────────
    with open(graded_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    total_correct = stage1_correct + stage2_correct
    print(f"\n{'='*60}")
    print(f"FINAL: {total_correct}/{total} correct ({100*total_correct/total:.1f}%)")
    print(f"  Stage 1: {stage1_correct}  |  Stage 2: {stage2_correct}  |  "
          f"No boxed: {no_boxed}  |  Bad format: {format_fail}  |  Errors: {errors}")
    print(f"Saved to {graded_path}")

    # Also write correct-only file
    correct_path = data_dir / "correct_only.jsonl"
    n_correct_written = 0
    with open(correct_path, "w") as f:
        for rec in results:
            if rec["correct"]:
                f.write(json.dumps(rec) + "\n")
                n_correct_written += 1
    print(f"Correct-only traces: {correct_path} ({n_correct_written} records)")


def main():
    parser = argparse.ArgumentParser(description="Grade generated reasoning traces")
    parser.add_argument("--judge-concurrency", type=int, default=10)
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge stage")
    parser.add_argument("--effortful", action="store_true",
                        help="Grade effortful traces from data_effortful/")
    parser.add_argument("--thinking", action="store_true",
                        help="Grade thinking model traces from data_thinking/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override data directory (must contain generations.jsonl)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

"""Smoke test: generate a few thinking traces with an arbitrary model and grade them.

Tests the full pipeline (generate → grade) on a small sample to verify
the model works with the thinking/reasoning API format.

Usage:
  python smoke_test_model.py --model moonshotai/kimi-k2.5
  python smoke_test_model.py --model moonshotai/kimi-k2.5 --num 5
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENROUTER_API_KEY, OPENROUTER_URL,
    THINKING_MODEL_MATH_PROMPT, THINKING_MODEL_CODE_PROMPT,
    THINKING_REASONING_TOKENS, THINKING_CONTENT_TOKENS,
    THINKING_QUESTIONS_PATH, THINKING_CODE_QUESTIONS_PATH,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_openrouter_reasoning, extract_code_from_response

# Math grading
sys.path.insert(0, "/workspace/tinker-cookbook")
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer


async def generate_one(session, semaphore, model, messages, reasoning_tokens, content_tokens):
    """Generate a single thinking trace."""
    reasoning, content = await call_openrouter_reasoning(
        session, model, messages,
        temperature=0.0, max_tokens=content_tokens,
        reasoning_max_tokens=reasoning_tokens,
        semaphore=semaphore,
        api_key=OPENROUTER_API_KEY, api_url=OPENROUTER_URL,
    )
    if str(reasoning).startswith("ERROR:"):
        return None, str(reasoning)
    response = f"<think>\n{reasoning}\n</think>\n\n{content}"
    return response, None


def grade_math(generation: str, ground_truth: str) -> tuple[bool, str]:
    """Quick math grading: extract boxed answer and check."""
    try:
        predicted = extract_boxed(generation)
    except Exception:
        return False, "(no \\boxed{} found)"
    try:
        correct = grade_answer(predicted, ground_truth)
        return correct, predicted
    except Exception:
        return False, predicted


def grade_code(generation: str, test_code: str) -> tuple[bool, str]:
    """Quick code grading: extract code and run tests."""
    import os, subprocess, tempfile
    code = extract_code_from_response(generation)
    if code is None:
        return False, "no code extracted"

    # Reuse the grade_code.py runner logic
    runner = '''
import sys, importlib.util, traceback
sys.path.insert(0, ".")
try:
    import solution
    spec = importlib.util.spec_from_file_location("test_solution", "test_solution.py")
    test_mod = importlib.util.module_from_spec(spec)
    for name in dir(solution):
        if not name.startswith('_'):
            setattr(test_mod, name, getattr(solution, name))
    spec.loader.exec_module(test_mod)
    test_funcs = [name for name in dir(test_mod) if name.startswith('test_')]
    failed = 0
    for name in test_funcs:
        try:
            getattr(test_mod, name)()
        except Exception:
            failed += 1
    if failed > 0 or len(test_funcs) == 0:
        sys.exit(1)
    else:
        sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(test_code)
        with open(os.path.join(tmpdir, "run_tests.py"), "w") as f:
            f.write(runner)
        try:
            result = subprocess.run(
                [sys.executable, "run_tests.py"],
                cwd=tmpdir, capture_output=True, timeout=15, text=True,
            )
            if result.returncode == 0:
                return True, "passed"
            return False, result.stderr.strip()[-200:] if result.stderr else "failed"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:100]


async def main():
    parser = argparse.ArgumentParser(description="Smoke test a thinking model")
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument("--num", type=int, default=10, help="Number of questions per task")
    parser.add_argument("--reasoning-tokens", type=int, default=THINKING_REASONING_TOKENS)
    parser.add_argument("--content-tokens", type=int, default=THINKING_CONTENT_TOKENS)
    args = parser.parse_args()

    print(f"Smoke test: {args.model}")
    print(f"  {args.num} math + {args.num} code questions")
    print(f"  reasoning_tokens={args.reasoning_tokens}, content_tokens={args.content_tokens}")
    print()

    # Load a few questions
    math_questions = []
    with open(THINKING_QUESTIONS_PATH) as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            math_questions.append(json.loads(line))

    code_questions = []
    with open(THINKING_CODE_QUESTIONS_PATH) as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            code_questions.append(json.loads(line))

    print(f"Loaded {len(math_questions)} math, {len(code_questions)} code questions")

    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession() as session:
        # ── Math ──────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"MATH ({len(math_questions)} questions)")
        print(f"{'='*60}")

        math_results = []
        for i, q in enumerate(math_questions):
            messages = [
                {"role": "system", "content": THINKING_MODEL_MATH_PROMPT},
                {"role": "user", "content": q["problem"]},
            ]
            t0 = time.time()
            gen, err = await generate_one(
                session, semaphore, args.model, messages,
                args.reasoning_tokens, args.content_tokens,
            )
            elapsed = time.time() - t0

            if err:
                print(f"  [{i+1}] ERROR: {err[:100]}")
                math_results.append({"correct": False, "error": err})
                continue

            correct, predicted = grade_math(gen, q["answer"])
            status = "CORRECT" if correct else "WRONG"
            think_len = len(gen.split("</think>")[0]) if "</think>" in gen else 0
            print(f"  [{i+1}] {status} (predicted={predicted}, expected={q['answer']}, "
                  f"think={think_len} chars, {elapsed:.1f}s)")
            math_results.append({"correct": correct, "predicted": predicted,
                                  "expected": q["answer"], "gen_length": len(gen)})

        math_acc = sum(1 for r in math_results if r.get("correct")) / len(math_results)
        print(f"\nMath accuracy: {math_acc*100:.0f}% ({sum(1 for r in math_results if r.get('correct'))}/{len(math_results)})")

        # ── Code ──────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"CODE ({len(code_questions)} questions)")
        print(f"{'='*60}")

        code_results = []
        for i, q in enumerate(code_questions):
            messages = [
                {"role": "system", "content": THINKING_MODEL_CODE_PROMPT},
                {"role": "user", "content": q["question"]},
            ]
            t0 = time.time()
            gen, err = await generate_one(
                session, semaphore, args.model, messages,
                args.reasoning_tokens, args.content_tokens,
            )
            elapsed = time.time() - t0

            if err:
                print(f"  [{i+1}] ERROR: {err[:100]}")
                code_results.append({"correct": False, "error": err})
                continue

            correct, detail = grade_code(gen, q["test"])
            status = "PASS" if correct else "FAIL"
            think_len = len(gen.split("</think>")[0]) if "</think>" in gen else 0
            print(f"  [{i+1}] {status} ({detail[:60]}, think={think_len} chars, {elapsed:.1f}s)")
            code_results.append({"correct": correct, "detail": detail,
                                  "gen_length": len(gen)})

        code_acc = sum(1 for r in code_results if r.get("correct")) / len(code_results)
        print(f"\nCode accuracy: {code_acc*100:.0f}% ({sum(1 for r in code_results if r.get('correct'))}/{len(code_results)})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"  Math: {math_acc*100:.0f}% ({len(math_results)} questions)")
    print(f"  Code: {code_acc*100:.0f}% ({len(code_results)} questions)")

    avg_math_len = sum(r.get("gen_length", 0) for r in math_results) / max(1, len(math_results))
    avg_code_len = sum(r.get("gen_length", 0) for r in code_results) / max(1, len(code_results))
    print(f"  Avg math generation length: {avg_math_len:.0f} chars")
    print(f"  Avg code generation length: {avg_code_len:.0f} chars")

    n_errors = sum(1 for r in math_results + code_results if "error" in r)
    if n_errors:
        print(f"  API errors: {n_errors}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

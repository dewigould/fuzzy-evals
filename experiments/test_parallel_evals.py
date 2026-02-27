"""Quick test: 4 base evals in parallel to verify semaphore + API throughput.

Runs 4 evals concurrently, each hitting 2 code datasets + 1 math dataset.
Code evals should funnel through the semaphore (max 2 concurrent).
Watch for "Acquiring code eval semaphore" messages interleaving correctly.

Usage:
    python run.py batch experiments/test_parallel_evals.py --max-parallel 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BaseModelEval, FullEvalConfig

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"

EXPERIMENTS = [
    BaseModelEval(
        name=f"stress_test_{i}",
        base_model=LLAMA_8B,
        eval=FullEvalConfig(
            datasets=("math_500", "mbppplus", "humanevalplus"),
            max_tokens=4096,
            max_problems=10,
        ),
    )
    for i in range(4)
]

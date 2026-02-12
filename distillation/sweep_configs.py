"""Hyperparameter sweep configurations for math and code distillation.

Qwen3-30B-A3B is MoE with only 3B active params, so LR can be slightly
higher than dense 8B models. Config A uses 2e-4 (aggressive), the rest
follow the standard pattern from prior Llama-8B SFT sweeps.
"""

MATH_CONFIGS = {
    "A_fast": {
        "lr": 2e-4,
        "steps": 500,
        "schedule": "linear",
        "batch_size": 50,
    },
    "B_medium": {
        "lr": 1e-4,
        "steps": 1000,
        "schedule": "linear",
        "batch_size": 50,
    },
    "C_gentle": {
        "lr": 5e-5,
        "steps": 1000,
        "schedule": "cosine",
        "batch_size": 50,
    },
    "D_long": {
        "lr": 5e-5,
        "steps": 2000,
        "schedule": "cosine",
        "batch_size": 50,
    },
}

CODE_CONFIGS = {
    "A_fast": {
        "lr": 2e-4,
        "steps": 500,
        "schedule": "linear",
        "batch_size": 50,
    },
    "B_medium": {
        "lr": 1e-4,
        "steps": 1000,
        "schedule": "linear",
        "batch_size": 50,
    },
    "C_gentle": {
        "lr": 5e-5,
        "steps": 1000,
        "schedule": "cosine",
        "batch_size": 50,
    },
    "D_long": {
        "lr": 5e-5,
        "steps": 2000,
        "schedule": "cosine",
        "batch_size": 50,
    },
}

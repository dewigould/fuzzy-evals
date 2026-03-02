# Fuzzy Evals

Studying distillation of reasoning traces into base models. We distill math and code reasoning traces from teacher models (Claude Sonnet 4.5, Kimi-K2.5, Qwen3-235B) into student models (Llama-8B, Llama-70B, Qwen-8B/30B) via LoRA SFT, then evaluate intra- and cross-domain performance across 11 benchmarks.

## Setup

```bash
# Clone
git clone <repo-url>
cd fuzzy-evals

# Python 3.13 venv
python3.13 -m venv fuzzy-evals-env
source fuzzy-evals-env/bin/activate
pip install -r requirements.txt  # if present, otherwise see CLAUDE.md for deps

# API keys — create /workspace/.env with:
#   OPENROUTER_API_KEY=...
#   TINKERY_API_KEY=...
#   HF_TOKEN=...

# Download trace data (private, requires HF access)
huggingface-cli login --token $HF_TOKEN
huggingface-cli download dewigould/fuzzy-evals-traces --repo-type dataset --local-dir traces/
```

## Quick Start

```bash
# Run a single training experiment
python run.py single --name math_sonnet --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --task math --traces sonnet --max-length 4096

# Evaluate a base model (no training)
python run.py base-eval --name qwen30b_base \
    --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Run a batch of experiments from a Python file
python run.py batch experiments/overnight_sweep.py --max-parallel 4

# Compare results across experiments
python run.py compare logs/2026-02-26_*

# Browse model responses interactively
streamlit run viewer.py --server.port 8501
```

## Data

Reasoning traces are stored on HuggingFace as a private dataset: [`dewigould/fuzzy-evals-traces`](https://huggingface.co/datasets/dewigould/fuzzy-evals-traces).

| File | Teacher | Task | Size |
|------|---------|------|------|
| `sonnet_math.jsonl` | Claude Sonnet 4.5 | math | 151 MB |
| `sonnet_code.jsonl` | Claude Sonnet 4.5 | code | 280 MB |
| `kimi_math.jsonl` | Kimi-K2.5 | math | 1.4 GB |
| `kimi_code.jsonl` | Kimi-K2.5 | code | 795 MB |
| `qwen_math.jsonl` | Qwen3-235B | math | 1.9 GB |
| `qwen_code.jsonl` | Qwen3-235B | code | 1.2 GB |

## Project Structure

See [CLAUDE.md](CLAUDE.md) for full architecture, config reference, and developer guide.

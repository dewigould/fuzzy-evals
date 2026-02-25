# Experiment Plan: RLVR vs SFT Training + Evaluation

## Context

Run a full experiment comparing 4 training approaches (Math RLVR, Code RLVR, Math SFT, Code SFT) on Llama-3.1-8B-Instruct, evaluating all models on MATH-500, BigCodeBench, philosophy, and weird questions. The repo has most scripts already written. Key changes needed: (1) fix formatting data generation to use 10+10+10, (2) sandbox BigCodeBench code execution with Docker, (3) restrict `<think>` seeding to fuzzy evals only.

---

## Phase 1: Environment Setup

### 1a. Install Python packages
```
pip3 install aiohttp pandas matplotlib numpy python-dotenv datasets chz tinker pyarrow
```

### 1b. Install tinker-cookbook
Clone to `/workspace/tinker-cookbook` and `pip install -e .` — all scripts expect it at that path via `sys.path.insert`.

### 1c. Fix Python 3.8 compatibility
`run_formatting_sft.py:37` uses `str | None` (requires 3.10+). Change to `Optional[str]`.

### 1d. Install Docker + build sandbox image
```bash
apt-get update && apt-get install -y docker.io
dockerd &  # RunPod may not use systemd
```
Create `/workspace/fuzzy-evals/Dockerfile.sandbox`:
```dockerfile
FROM python:3.10-slim
RUN pip install --no-cache-dir numpy pandas scipy matplotlib sympy requests \
    beautifulsoup4 lxml regex python-dateutil pytz pillow openpyxl \
    networkx scikit-learn cryptography
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /home/sandbox
```
Build: `docker build -t code-sandbox -f Dockerfile.sandbox .`

### 1e. Verify `.env`
Confirm `/workspace/.env` has `OPENROUTER_API_KEY`.

---

## Phase 2: Code Changes

### 2a. `run_generate_formatting_data.py` — Change to 10+10+10
**File**: `/workspace/fuzzy-evals/scripts/run_generate_formatting_data.py`

- **Math source**: Replace Hendrycks MATH algebra train with first 10 MATH-500 problems:
  ```python
  math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
  math_problems = [math500[i] for i in range(10)]
  ```
- **Code source**: Change from BigCodeBench indices 500-549 to indices 0-9:
  ```python
  code_tasks = [code_ds[i] for i in range(10)]
  ```
- **Add weird questions**: Load first 10 from `weird_questions.json`, generate responses via Sonnet 3.5:
  ```python
  with open('/workspace/fuzzy-evals/dataset_jsons/weird_questions.json') as f:
      weird_data = json.load(f)
  weird_questions = weird_data[:10]
  ```
  Generate with prompt: "Answer the following question thoughtfully and in detail."
- Update all loop sizes from 50 to 10

### 2b. `run_formatting_sft.py` — Adjust batch size
**File**: `/workspace/fuzzy-evals/scripts/run_formatting_sft.py`

- Line 58: Change `batch_size=10` to `batch_size=3` (30 examples / 3 = 10 steps)

### 2c. `run_eval_sweep.py` — Docker sandboxing for BigCodeBench
**File**: `/workspace/fuzzy-evals/scripts/run_eval_sweep.py`

Replace `run_test_in_subprocess()` (lines 218-263) with `run_test_in_docker()`:
- Run each test in a Docker container with:
  - `--memory=512m --memory-swap=512m` (prevents OOM killing the host)
  - `--cpus=1`
  - `--network=none` (no network access)
  - `--read-only` + `--tmpfs /tmp:size=64m`
  - `--rm` (auto-cleanup)
- Pipe runner code to `python3` via stdin inside container
- Reduce ThreadPoolExecutor to 4 workers (line 381)
- This directly addresses the RunPod OOM kills — untrusted code must run with memory limits and limited parallelism to stay within 60GB

### 2d. `run_eval_sweep.py` — Think seeding only for fuzzy evals
**File**: `/workspace/fuzzy-evals/scripts/run_eval_sweep.py`

Modify the main eval loop (lines 617-649):
- Skip math and code evals for seeded model keys (`base_seeded`, `math_sft_seeded`, `code_sft_seeded`)
- Only run fuzzy evals (philosophy + weird_questions) for seeded variants
- Keep `SEEDED_MODELS` dict unchanged — the seeded keys still get `<think>` prefix via `sample_tinker()`

---

## Phase 3: Execution Order

```
Step 1: Environment setup (Phase 1)
    |
Step 2: Code changes (Phase 2)
    |
Step 3: Training (sequential — one at a time, ~2GB RAM each)
    |-- run_math_sft.py   (500 steps) — DONE
    |-- run_code_sft.py   (500 steps) — DONE
    |-- run_math_rlvr.py  (500 steps) — resumes from step 200
    |-- run_code_rlvr.py  (500 steps) — resumes from step 100
    |
    |-- run_generate_formatting_data.py  (can run alongside training — only calls OpenRouter) — DONE
    |
Step 4: run_formatting_sft.py  (requires: Step 3 complete)
    |   Applies 10 steps to each of: base, math_rlvr, code_rlvr, math_sft, code_sft
    |
Step 5: run_eval_sweep.py  (requires: Step 4 complete)
    |   8 model variants x 4 domains
    |   Math/Code: only 5 base models
    |   Fuzzy: 5 base + 3 seeded = 8 variants
```

---

## Phase 4: Verification

After each step, verify:
1. **Env**: `python3 -c "import tinker, tinker_cookbook; print('OK')"` + `docker run --rm code-sandbox python3 -c "print('OK')"`
2. **Training**: Check `checkpoints.jsonl` exists for all 4 models
3. **Formatting data**: `wc -l formatting_combined.jsonl` = ~30 lines
4. **Formatting SFT**: `checkpoints.jsonl` exists for all 5 format_sft models
5. **Eval**: `eval_scores.parquet` exists, verify model/dataset counts with pandas

---

## Files Modified
| File | Change |
|------|--------|
| `scripts/run_generate_formatting_data.py` | 10+10+10 instead of 50+50 |
| `scripts/run_formatting_sft.py` | batch_size=3, fix `str\|None` syntax |
| `scripts/run_eval_sweep.py` | Docker sandbox + seeded-only-fuzzy logic |

## Files Created
| File | Purpose |
|------|---------|
| `Dockerfile.sandbox` | Lightweight Python image for code execution |
| `run_all.sh` | Orchestration script for the full pipeline |

---

## Memory Management Strategy (Detailed)

### The problem
The pod has **60GB RAM** (64GB cgroup limit), **no swap**, and **no ulimits** — all resource limits are `unlimited`. The root filesystem is only 20GB overlay. RunPod enforces the memory limit via cgroups (`/sys/fs/cgroup/memory.max = 64000000000`). Exceeding this triggers the OOM killer. There are multiple memory pressure sources:

### Source 1: BigCodeBench code execution (PRIMARY SUSPECT)
**Current**: `run_test_in_subprocess()` spawns Python subprocesses running untrusted LLM-generated code. Without memory limits, a single buggy test that allocates a large array (e.g., `[0] * 10**9`) would consume ~8GB. With multiple workers, this quickly exceeds the 60GB limit and triggers the OOM killer. **This was the confirmed cause of the OOM crash.**

**Fix**: Docker containers with hard memory caps:
```
docker run --rm --memory=512m --memory-swap=512m --pids-limit=64 ...
```
- **512MB per container** x 4 workers max = **2GB worst case** (vs unlimited before)
- `--memory-swap=512m` prevents swap usage (there's no swap anyway, but this prevents the kernel from overcommitting)
- `--pids-limit=64` prevents fork bombs
- If a container exceeds 512MB, Docker kills it (returns exit code 137) — the test is marked as failed, not the whole process
- **Note**: `run_code_rlvr.py` also needs RLIMIT_AS=512MB on its subprocess calls (applied via resource_wrapper)

### Source 2: HuggingFace dataset caching
**Problem**: By default, HF `datasets` caches downloaded data to `~/.cache/huggingface/`. The root filesystem is only 20GB. Large datasets like OpenR1-Math-220k (220K examples) or BigCodeBench could fill this.

**Fix**: Set `HF_HOME=/workspace/.cache/huggingface` to redirect cache to the 873TB network storage. Add to all scripts and to `.env`:
```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
```
Most training scripts already use `streaming=True` which avoids full download, but eval datasets (MATH-500, BigCodeBench) are loaded in full.

### Source 3: Eval sweep accumulating results in memory
**Current**: `run_eval_sweep.py` appends all results to `all_results` list in memory and periodically writes to parquet. With 8 models x 4 domains x 500+ questions, the list grows large, especially since it stores full `raw_output` text (model completions can be ~2KB each).

**Fix**:
- Add periodic `gc.collect()` after each model x domain evaluation
- The existing checkpoint/resume mechanism (writing `eval_partial.parquet` after each domain) already provides robustness — if the process is killed, it can resume from the last completed (model, dataset) pair
- Estimate: 8,920 results x ~3KB average = ~27MB — this is actually small. The bigger issue is holding all async tasks and responses in memory simultaneously during fuzzy eval.

### Source 4: Async task accumulation in eval_fuzzy()
**Current**: Lines 409-410 create all fuzzy generation tasks upfront, then `asyncio.gather()` holds all responses in memory. For weird_questions: 46Q x 10S = 460 completions, each potentially 2-4KB = ~2MB. This is manageable.

**Fix**: Already bounded by semaphore (15 concurrent). No change needed, but add `gc.collect()` between philosophy and weird_questions evaluation.

### Source 5: Docker image consuming root disk
**Problem**: Root filesystem is only 20GB with 19GB free. A Docker image with scientific Python packages could be 1-2GB.

**Fix**:
- Keep the Docker image minimal (~500MB target)
- Store Docker data directory on /workspace if needed: `dockerd --data-root /workspace/.docker &`
- Clean up unused images after build: `docker image prune -f`

### Comprehensive memory safeguards in `run_eval_sweep.py`:

```python
import gc
import resource

# 1. Set process-level memory limit as safety net (50GB of 60GB)
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, hard))
except:
    pass  # May not work in all environments

# 2. After each model x domain eval:
gc.collect()

# 3. Docker containers get --memory=512m (handled in run_test_in_docker)

# 4. Monitor memory before each major section:
def check_memory():
    """Check cgroup memory (not host /proc/meminfo which shows the full host)."""
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            current = int(f.read().strip())
        limit = 64_000_000_000  # 60GB cgroup limit
        pct = current / limit * 100
        if pct > 80:
            log.warning(f"HIGH MEMORY: {pct:.1f}% ({current/1024**3:.1f}GB / {limit/1024**3:.1f}GB)")
            gc.collect()
        return pct
    except Exception:
        return 0
```

### Crash recovery: the experiments WILL finish

The key guarantee is **checkpoint/resume at every level**:

1. **Training scripts**: All 4 training scripts use tinker's built-in `save_every=50` checkpoints. If a training run is killed, re-running the same script resumes from the last checkpoint via `checkpoint_utils.get_last_checkpoint()`.

2. **Formatting data generation**: The output is a single JSONL file. If killed mid-generation, re-run regenerates everything (~30 API calls, takes minutes). We could add incremental writing but it's not worth the complexity for 30 examples.

3. **Formatting SFT**: Already skips completed models (line 90-92 checks for `ckpt_file.exists()`). If killed after 3 of 5 models, re-run picks up the remaining 2.

4. **Eval sweep** (most critical):
   - Writes `eval_partial.parquet` after every (model, dataset) pair completes
   - On restart, loads partial results and skips completed pairs (lines 607-614)
   - With 8 models x 4 domains = up to 32 checkpoints
   - Even if killed during BigCodeBench eval for model 3, it only needs to redo that one model x code evaluation, not everything

5. **Orchestration script** (`run_all.sh`):
   - Check exit codes after each phase
   - On failure: log which phase failed, allow manual restart from that phase
   - Add a `--resume` flag that skips completed phases by checking for output artifacts

### Specific `run_all.sh` resilience pattern:

```bash
#!/bin/bash
set -euo pipefail

# Memory monitoring in background
monitor_memory() {
    while true; do
        MEM_PCT=$(free | awk '/Mem:/{printf "%.0f", $3/$2*100}')
        if [ "$MEM_PCT" -gt 85 ]; then
            echo "[MEMORY WARNING] ${MEM_PCT}% used at $(date)"
        fi
        sleep 30
    done
}
monitor_memory &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Phase 1: Training (with retry)
run_with_retry() {
    local script=$1
    local log_file=$2
    local max_retries=3
    for i in $(seq 1 $max_retries); do
        echo "Attempt $i/$max_retries: $script"
        if python3 "$script" >> "$log_file" 2>&1; then
            echo "SUCCESS: $script"
            return 0
        fi
        echo "FAILED attempt $i. Retrying in 30s..."
        sleep 30
    done
    echo "FAILED after $max_retries attempts: $script"
    return 1
}
```

---

## Risk: Docker-in-Docker on RunPod

RunPod pods may not support Docker. If Docker installation fails, **fallback: process-level resource limits**:

```python
def run_test_sandboxed(code, test_code, entry_point, timeout=30):
    """Fallback: subprocess with resource limits (no Docker)."""
    # Create wrapper that sets resource limits inside the child process
    wrapper = f'''
import resource, signal, sys, os
# Memory limit: 512MB
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
# CPU time limit: {timeout} seconds
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
# Max file size: 10MB (prevent disk filling)
resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
# Max processes: 32 (prevent fork bombs)
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
signal.alarm({timeout})
'''
    runner_code = wrapper + "\n" + <existing runner_code>

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write and run in isolated tmpdir
        tmppath = os.path.join(tmpdir, 'run.py')
        with open(tmppath, 'w') as f:
            f.write(runner_code)
        result = subprocess.run(
            [sys.executable, tmppath],
            capture_output=True, text=True,
            timeout=timeout + 5,
            cwd=tmpdir,  # Isolate working directory
        )
```

This gives us:
- **RLIMIT_AS=512MB**: Hard memory cap per test process
- **RLIMIT_CPU=30s**: Hard CPU time cap
- **RLIMIT_FSIZE=10MB**: Prevents filling disk
- **RLIMIT_NPROC=32**: Prevents fork bombs
- **signal.alarm()**: Wall-clock timeout
- **tmpdir isolation**: Clean working directory

Both Docker and fallback approaches guarantee the experiments can finish even if individual tests try to consume excessive resources.

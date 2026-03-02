"""Streamlit viewer for browsing model responses in experiment parquet files."""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

LOGS_DIR = Path(__file__).parent / "logs"


def find_experiments() -> dict[str, Path]:
    """Scan logs/ for experiment dirs containing parquet files.

    Returns {display_name: path} dict. Handles both flat logs/ and
    nested batch dirs like logs/overnight_sweep/exp_name/.
    """
    experiments = {}
    if not LOGS_DIR.exists():
        return experiments

    for batch_dir in sorted(LOGS_DIR.iterdir()):
        if not batch_dir.is_dir():
            continue
        if _is_experiment_dir(batch_dir):
            experiments[batch_dir.name] = batch_dir
            continue
        # Batch directory — scan subdirs
        for exp_dir in sorted(batch_dir.iterdir()):
            if exp_dir.is_dir() and _is_experiment_dir(exp_dir):
                display = f"{batch_dir.name}/{exp_dir.name}"
                experiments[display] = exp_dir
    return experiments


def _is_experiment_dir(p: Path) -> bool:
    return (p / "full_eval").is_dir() or (p / "checkpoint_eval").is_dir()


def find_eval_phases(exp_dir: Path) -> list[str]:
    """List available eval phases: 'full_eval' and/or checkpoint steps."""
    phases = []
    if (exp_dir / "full_eval").is_dir() and any(
        (exp_dir / "full_eval").glob("*.parquet")
    ):
        phases.append("full_eval")
    ckpt_dir = exp_dir / "checkpoint_eval"
    if ckpt_dir.is_dir():
        steps = sorted(
            [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
            key=lambda s: int(s.replace("step", "")),
        )
        for step in steps:
            if any((ckpt_dir / step).glob("*.parquet")):
                phases.append(f"checkpoint_eval/{step}")
    return phases


def find_datasets(exp_dir: Path, phase: str) -> list[str]:
    """List parquet dataset names in a given phase dir."""
    phase_dir = exp_dir / phase
    if not phase_dir.is_dir():
        return []
    files = sorted(phase_dir.glob("results_*.parquet"))
    return [f.stem.replace("results_", "") for f in files]


def load_parquet(exp_dir: Path, phase: str, dataset: str) -> pd.DataFrame:
    path = exp_dir / phase / f"results_{dataset}.parquet"
    return pd.read_parquet(path)


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect math/code/fuzzy based on columns."""
    if "correct" in df.columns:
        return "math"
    if "passed" in df.columns:
        return "code"
    if "score" in df.columns:
        return "fuzzy"
    return "unknown"


def _get(row, col, default=""):
    """Safely get a column value, returning default if missing or NaN."""
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return val


def main():
    st.set_page_config(page_title="Response Viewer", layout="wide")
    st.title("Response Viewer")

    # --- Sidebar ---
    experiments = find_experiments()
    if not experiments:
        st.error("No experiments found in logs/. Run some experiments first.")
        return

    with st.sidebar:
        st.header("Navigation")

        exp_name = st.selectbox("Experiment", list(experiments.keys()))
        exp_dir = experiments[exp_name]

        phases = find_eval_phases(exp_dir)
        if not phases:
            st.warning("No eval results found for this experiment.")
            return
        phase = st.selectbox("Eval Phase", phases)

        datasets = find_datasets(exp_dir, phase)
        if not datasets:
            st.warning("No parquet files in this phase.")
            return
        dataset = st.selectbox("Dataset", datasets)

        # Filter
        filter_opt = st.radio("Filter", ["All", "Correct only", "Incorrect only"], horizontal=True)

        st.divider()

    # --- Load data ---
    df = load_parquet(exp_dir, phase, dataset)
    dtype = detect_dataset_type(df)

    # Determine the boolean column
    if dtype == "math":
        bool_col = "correct"
    elif dtype == "code":
        bool_col = "passed"
    else:
        bool_col = None

    # Apply filter
    if filter_opt == "Correct only" and bool_col:
        df = df[df[bool_col] == True].reset_index(drop=True)
    elif filter_opt == "Incorrect only" and bool_col:
        df = df[df[bool_col] == False].reset_index(drop=True)

    total = len(df)
    if total == 0:
        st.info("No problems match the current filter.")
        return

    # --- Navigation ---
    nav_key = f"{exp_name}|{phase}|{dataset}|{filter_opt}"
    if "nav_key" not in st.session_state or st.session_state.nav_key != nav_key:
        st.session_state.nav_key = nav_key
        st.session_state.idx = 0

    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Prev", use_container_width=True):
                st.session_state.idx = max(0, st.session_state.idx - 1)
        with col2:
            if st.button("Next", use_container_width=True):
                st.session_state.idx = min(total - 1, st.session_state.idx + 1)

        idx = st.number_input(
            "Go to index",
            min_value=0,
            max_value=total - 1,
            value=st.session_state.idx,
            step=1,
            key="idx_input",
        )
        st.session_state.idx = idx

    idx = st.session_state.idx
    row = df.iloc[idx]

    # --- Stats bar ---
    if bool_col:
        n_correct = int(df[bool_col].sum())
        pct = n_correct / total * 100 if total else 0
        label = "correct" if dtype == "math" else "passed"
        st.markdown(
            f"**Problem {idx + 1} of {total}** &nbsp;|&nbsp; "
            f"**{n_correct} {label} ({pct:.1f}%)**"
        )
    else:
        st.markdown(f"**Problem {idx + 1} of {total}**")

    st.divider()

    # --- Render based on dataset type ---
    if dtype == "math":
        render_math(row)
    elif dtype == "code":
        render_code(row)
    elif dtype == "fuzzy":
        render_fuzzy(row)
    else:
        render_unknown(row)


def render_math(row):
    correct = row.get("correct", None)
    if correct is True:
        st.success("Correct")
    elif correct is False:
        st.error("Incorrect")

    # Question
    st.subheader("Problem")
    st.markdown(_get(row, "question", "N/A"))

    # Answers side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ground Truth**")
        st.code(str(_get(row, "ground_truth", "N/A")), language=None)
    with col2:
        st.markdown("**Predicted Answer**")
        st.code(str(_get(row, "predicted_answer", "N/A")), language=None)

    # Grading method if present
    method = _get(row, "grading_method", "")
    if method:
        st.caption(f"Grading: {method}")

    # Chain of thought
    cot = _get(row, "cot", "")
    if cot:
        with st.expander("Thinking / Chain of Thought"):
            st.markdown(str(cot))

    # Final answer portion
    user_output = _get(row, "user_output", "")
    if user_output:
        st.subheader("Model Output")
        st.markdown(str(user_output))

    # Response length
    raw = str(_get(row, "raw_output", ""))
    st.caption(f"Response length: {len(raw):,} chars")

    # Full raw output
    with st.expander("Full Raw Response"):
        st.text(raw)


def render_code(row):
    passed = row.get("passed", None)
    if passed is True:
        st.success("Passed")
    elif passed is False:
        st.error("Failed")

    # Problem / task ID
    st.subheader("Problem")
    question = _get(row, "question", "")
    task_id = _get(row, "task_id", "")
    func_name = _get(row, "function_name", "")
    if question:
        st.markdown(str(question))
    elif task_id:
        label = f"`{task_id}`"
        if func_name:
            label += f" — `{func_name}`"
        st.markdown(label)

    # Chain of thought
    cot = _get(row, "cot", "")
    if cot:
        with st.expander("Thinking / Chain of Thought"):
            st.markdown(str(cot))

    # Code output
    user_output = _get(row, "user_output", "")
    if user_output:
        st.subheader("Extracted Code")
        st.code(str(user_output), language="python")

    # Test detail
    detail = _get(row, "detail", "")
    if detail:
        with st.expander("Test Detail"):
            st.text(str(detail))

    # Response length
    raw = str(_get(row, "raw_output", ""))
    st.caption(f"Response length: {len(raw):,} chars")

    # Full raw output
    with st.expander("Full Raw Response"):
        st.text(raw)


def render_fuzzy(row):
    score = row.get("score", None)
    if score is not None and not pd.isna(score):
        st.metric("Score", f"{score}/48")

    st.subheader("Question")
    st.markdown(_get(row, "question", "N/A"))

    cot = _get(row, "cot", "")
    if cot:
        with st.expander("Thinking / Chain of Thought"):
            st.markdown(str(cot))

    user_output = _get(row, "user_output", "")
    if user_output:
        st.subheader("Model Response")
        st.markdown(str(user_output))

    judge = _get(row, "judge_response", "")
    if judge:
        with st.expander("Judge Response"):
            st.markdown(str(judge))

    raw = str(_get(row, "raw_output", _get(row, "full_response", "")))
    st.caption(f"Response length: {len(raw):,} chars")


def render_unknown(row):
    st.warning("Unknown dataset type -- showing all columns.")
    for col in row.index:
        val = row[col]
        st.subheader(col)
        if isinstance(val, str) and len(val) > 500:
            with st.expander(f"{col} (long)"):
                st.text(val)
        else:
            st.write(val)


if __name__ == "__main__":
    main()

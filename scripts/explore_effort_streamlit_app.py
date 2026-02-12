"""
Streamlit app for qualitative exploration of Opus 4.6 effort sweep results.

Run:  conda activate fuzzy-evals && streamlit run scripts/explore_effort_streamlit_app.py
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "results_opus_sweep")
DATASETS_DIR = os.path.join(BASE_DIR, "dataset_jsons")

# ── Data loading (cached) ──────────────────────────────────────────────

@st.cache_data
def load_data():
    # 1) Scores
    scores = pd.read_parquet(os.path.join(RESULTS_DIR, "effort_scores.parquet"))

    # 2) Raw outputs  (JSONL → dict keyed by (condition, dataset, question_id, sample_id))
    raw = {}
    with open(os.path.join(RESULTS_DIR, "raw_outputs_effort.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["condition"], rec["dataset"], rec["question_id"], rec["sample_id"])
            raw[key] = rec

    # 3) Questions
    questions = {}
    # Philosophy
    with open(os.path.join(DATASETS_DIR, "philosophy_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("philosophy", i)] = q["question"]
    # Weird questions
    with open(os.path.join(DATASETS_DIR, "weird_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("weird_questions", i)] = q["prompt"]
    # Futuristic tech
    with open(os.path.join(DATASETS_DIR, "futuristic_tech_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("futuristic_tech", i)] = q["question"]

    # 4) Attach raw text + question text to scores df
    raw_outputs = []
    judge_responses = []
    question_texts = []
    for _, row in scores.iterrows():
        condition = f"effort_{row['effort_level']}"
        key = (condition, row["dataset"], row["question_id"], row["sample_id"])
        rec = raw.get(key, {})
        raw_outputs.append(rec.get("raw_output", ""))
        judge_responses.append(rec.get("judge_response", ""))
        question_texts.append(questions.get((row["dataset"], row["question_id"]), ""))

    scores = scores.copy()
    scores["raw_output"] = raw_outputs
    scores["judge_response"] = judge_responses
    scores["question_text"] = question_texts

    return scores


# ── Rubric criteria per dataset ────────────────────────────────────────

CRITERIA = {
    "philosophy": [
        "thesis_clarity", "argumentative_soundness", "dialectical_engagement",
        "precision_distinctions", "substantive_contribution", "example_quality",
    ],
    "weird_questions": [
        "willingness_to_engage", "specificity_concreteness", "reasoning_depth",
        "intellectual_risk_taking", "creative_insight", "epistemic_calibration",
    ],
    "futuristic_tech": [
        "scientific_grounding", "bottleneck_identification", "timeline_plausibility",
        "scaling_deployment_realism", "integration_awareness", "epistemic_honesty",
    ],
}


# ── App ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Effort Sweep Explorer", layout="wide")
    st.title("Opus 4.6 Effort Sweep — Qualitative Explorer")

    df = load_data()

    # ── Sidebar filters ────────────────────────────────────────────────
    st.sidebar.header("Filters")

    # Dataset
    datasets = sorted(df["dataset"].unique())
    sel_dataset = st.sidebar.selectbox("Dataset", datasets)

    # Question
    sub = df[df["dataset"] == sel_dataset]
    q_ids = sorted(sub["question_id"].unique())
    q_labels = {}
    for qid in q_ids:
        txt = sub[sub["question_id"] == qid]["question_text"].iloc[0]
        short = txt[:80] + "…" if len(txt) > 80 else txt
        q_labels[qid] = f"Q{qid}: {short}"
    sel_qid = st.sidebar.selectbox("Question", q_ids, format_func=lambda x: q_labels[x])

    # Effort levels
    all_efforts = ["low", "medium", "high", "max"]
    sel_efforts = st.sidebar.multiselect("Effort levels", all_efforts, default=all_efforts)

    # ── Filter data ────────────────────────────────────────────────────
    mask = (
        (df["dataset"] == sel_dataset)
        & (df["question_id"] == sel_qid)
        & (df["effort_level"].isin(sel_efforts))
    )
    filtered = df[mask].copy()

    # ── Question display ───────────────────────────────────────────────
    st.subheader("Question")
    if len(filtered) > 0:
        st.markdown(f"> {filtered['question_text'].iloc[0]}")
    else:
        st.warning("No data for this selection.")
        return

    # ── Score overview ─────────────────────────────────────────────────
    st.subheader("Score Distribution by Effort Level")

    valid = filtered.dropna(subset=["score"])

    # Summary table
    summary = (
        valid.groupby("effort_level")["score"]
        .agg(["count", "mean", "std", "min", "max"])
        .reindex([e for e in all_efforts if e in sel_efforts])
    )
    summary.columns = ["N (valid)", "Mean", "Std", "Min", "Max"]
    st.dataframe(summary.style.format({"Mean": "{:.1f}", "Std": "{:.1f}", "Min": "{:.0f}", "Max": "{:.0f}"}))

    # Criteria breakdown
    criteria = CRITERIA.get(sel_dataset, [])
    if criteria:
        crit_means = valid.groupby("effort_level")[criteria].mean().reindex(
            [e for e in all_efforts if e in sel_efforts]
        )
        st.markdown("**Mean criteria scores (0-8 each):**")
        st.dataframe(crit_means.style.format("{:.1f}"))

    # ── Score scatter / strip chart ────────────────────────────────────
    st.subheader("All Samples")

    # Build a display table
    display_cols = ["effort_level", "sample_id", "score", "reasoning_tokens", "output_tokens"]
    display = valid[display_cols].sort_values(["effort_level", "score"], ascending=[True, False])
    st.dataframe(display.reset_index(drop=True), height=300)

    # ── Interesting pairs: high vs low scores ──────────────────────────
    st.subheader("Interesting Comparisons")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pick a HIGH-scoring response:**")
        high_sorted = valid.sort_values("score", ascending=False)
        high_options = []
        for _, row in high_sorted.iterrows():
            high_options.append(
                f"[{row['effort_level']}] sample {int(row['sample_id'])} — score {row['score']:.0f}"
            )
        if high_options:
            sel_high = st.selectbox("High scorer", range(len(high_options)),
                                     format_func=lambda i: high_options[i], key="high")
            high_row = high_sorted.iloc[sel_high]

    with col2:
        st.markdown("**Pick a LOW-scoring response:**")
        low_sorted = valid.sort_values("score", ascending=True)
        low_options = []
        for _, row in low_sorted.iterrows():
            low_options.append(
                f"[{row['effort_level']}] sample {int(row['sample_id'])} — score {row['score']:.0f}"
            )
        if low_options:
            sel_low = st.selectbox("Low scorer", range(len(low_options)),
                                    format_func=lambda i: low_options[i], key="low")
            low_row = low_sorted.iloc[sel_low]

    # ── Side-by-side response display ──────────────────────────────────
    if high_options and low_options:
        st.divider()
        st.subheader("Side-by-Side Comparison")

        col_a, col_b = st.columns(2)

        for col, row, label in [(col_a, high_row, "HIGH"), (col_b, low_row, "LOW")]:
            with col:
                score_val = row["score"]
                st.markdown(
                    f"### {label}: score **{score_val:.0f}** "
                    f"(effort=`{row['effort_level']}`, sample={int(row['sample_id'])})"
                )

                # Token info
                st.caption(
                    f"reasoning_tokens={int(row.get('reasoning_tokens', 0)):,}  |  "
                    f"output_tokens={int(row.get('output_tokens', 0)):,}"
                )

                # Criteria breakdown
                if criteria:
                    crit_vals = {c: row[c] for c in criteria if pd.notna(row[c])}
                    if crit_vals:
                        st.markdown("**Criteria:**")
                        cols_inner = st.columns(min(len(crit_vals), 3))
                        for i, (c, v) in enumerate(crit_vals.items()):
                            cols_inner[i % 3].metric(c.replace("_", " ").title(), f"{v:.0f}/8")

                # Model response
                st.markdown("**Model Response:**")
                output = row.get("raw_output", "")
                if output:
                    with st.expander("Show full response", expanded=True):
                        st.markdown(output)
                else:
                    st.warning("Raw output not found")

                # Judge response
                st.markdown("**Judge Feedback:**")
                judge = row.get("judge_response", "")
                if judge:
                    with st.expander("Show judge response", expanded=False):
                        st.code(judge, language="json")

    # ── Browse all responses ───────────────────────────────────────────
    st.divider()
    st.subheader("Browse Individual Responses")

    browse_effort = st.selectbox(
        "Effort level",
        [e for e in all_efforts if e in sel_efforts],
        key="browse_effort",
    )
    browse_sub = valid[valid["effort_level"] == browse_effort].sort_values("score", ascending=False)
    browse_options = []
    for _, row in browse_sub.iterrows():
        browse_options.append(
            f"sample {int(row['sample_id'])} — score {row['score']:.0f}"
        )
    if browse_options:
        sel_browse = st.selectbox("Sample", range(len(browse_options)),
                                   format_func=lambda i: browse_options[i], key="browse_sample")
        brow_row = browse_sub.iloc[sel_browse]

        st.markdown(f"**Score: {brow_row['score']:.0f}** | "
                    f"reasoning_tokens={int(brow_row.get('reasoning_tokens', 0)):,} | "
                    f"output_tokens={int(brow_row.get('output_tokens', 0)):,}")

        if criteria:
            crit_vals = {c: brow_row[c] for c in criteria if pd.notna(brow_row[c])}
            if crit_vals:
                cols_c = st.columns(min(len(crit_vals), 6))
                for i, (c, v) in enumerate(crit_vals.items()):
                    cols_c[i % 6].metric(c.replace("_", " ").title(), f"{v:.0f}/8")

        output = brow_row.get("raw_output", "")
        if output:
            st.markdown(output)
        else:
            st.warning("Raw output not found")

        judge = brow_row.get("judge_response", "")
        if judge:
            with st.expander("Judge response"):
                st.code(judge, language="json")


if __name__ == "__main__":
    main()

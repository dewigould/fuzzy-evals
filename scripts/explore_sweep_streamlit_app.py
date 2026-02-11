"""
Streamlit app for qualitative exploration of Opus 4.6 prompt-style sweep results.
Compare responses across different prompt styles for the same question.

Run:  conda activate fuzzy-evals && streamlit run scripts/explore_sweep_streamlit_app.py
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

ALL_PROMPT_STYLES = ["minimal", "medium", "detailed", "think_hard", "think_hard_rubric"]

STYLE_LABELS = {
    "minimal": "Minimal",
    "medium": "Medium",
    "detailed": "Detailed",
    "think_hard": "Think Hard",
    "think_hard_rubric": "Think Hard + Rubric",
}

# ── Data loading (cached) ──────────────────────────────────────────────

@st.cache_data
def load_data():
    # 1) Scores
    scores = pd.read_parquet(os.path.join(RESULTS_DIR, "sweep_scores.parquet"))

    # 2) Raw outputs (JSONL → dict keyed by (condition, dataset, question_id, sample_id))
    raw = {}
    with open(os.path.join(RESULTS_DIR, "raw_outputs_sweep.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["condition"], rec["dataset"], rec["question_id"], rec["sample_id"])
            raw[key] = rec

    # 3) Questions
    questions = {}
    with open(os.path.join(DATASETS_DIR, "philosophy_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("philosophy", i)] = q["question"]
    with open(os.path.join(DATASETS_DIR, "weird_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("weird_questions", i)] = q["prompt"]
    with open(os.path.join(DATASETS_DIR, "futuristic_tech_questions.json")) as f:
        for i, q in enumerate(json.load(f)):
            questions[("futuristic_tech", i)] = q["question"]

    # 4) Attach raw text + question text to scores df
    raw_outputs = []
    judge_responses = []
    question_texts = []
    for _, row in scores.iterrows():
        condition = f"sweep_{row['prompt_style']}"
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


# ── App ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Prompt Sweep Explorer", layout="wide")
    st.title("Opus 4.6 Prompt-Style Sweep — Qualitative Explorer")

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

    # Prompt styles
    available_styles = sorted(df["prompt_style"].unique(), key=lambda s: ALL_PROMPT_STYLES.index(s) if s in ALL_PROMPT_STYLES else 99)
    sel_styles = st.sidebar.multiselect(
        "Prompt styles",
        available_styles,
        default=available_styles,
        format_func=lambda s: STYLE_LABELS.get(s, s),
    )

    if not sel_styles:
        st.warning("Select at least one prompt style.")
        return

    # ── Filter data ────────────────────────────────────────────────────
    mask = (
        (df["dataset"] == sel_dataset)
        & (df["question_id"] == sel_qid)
        & (df["prompt_style"].isin(sel_styles))
    )
    filtered = df[mask].copy()

    # ── Question display ───────────────────────────────────────────────
    st.subheader("Question")
    if len(filtered) > 0:
        st.markdown(f"> {filtered['question_text'].iloc[0]}")
    else:
        st.warning("No data for this selection.")
        return

    criteria = CRITERIA.get(sel_dataset, [])

    # ── Score overview ─────────────────────────────────────────────────
    st.subheader("Score Distribution by Prompt Style")

    valid = filtered.dropna(subset=["score"])

    # Summary table
    style_order = [s for s in ALL_PROMPT_STYLES if s in sel_styles]
    summary = (
        valid.groupby("prompt_style")["score"]
        .agg(["count", "mean", "std", "min", "max"])
        .reindex(style_order)
    )
    summary.columns = ["N (valid)", "Mean", "Std", "Min", "Max"]
    summary.index = summary.index.map(lambda s: STYLE_LABELS.get(s, s))
    st.dataframe(summary.style.format({"Mean": "{:.1f}", "Std": "{:.1f}", "Min": "{:.0f}", "Max": "{:.0f}"}))

    # Criteria breakdown
    if criteria:
        crit_means = (
            valid.groupby("prompt_style")[criteria]
            .mean()
            .reindex(style_order)
        )
        crit_means.index = crit_means.index.map(lambda s: STYLE_LABELS.get(s, s))
        st.markdown("**Mean criteria scores:**")
        st.dataframe(crit_means.style.format("{:.1f}"))

    # Token usage breakdown
    token_means = (
        valid.groupby("prompt_style")[["reasoning_tokens", "output_tokens"]]
        .mean()
        .reindex(style_order)
    )
    token_means.index = token_means.index.map(lambda s: STYLE_LABELS.get(s, s))
    st.markdown("**Mean token usage:**")
    st.dataframe(token_means.style.format("{:,.0f}"))

    # ── Side-by-side prompt style comparison ───────────────────────────
    st.divider()
    st.subheader("Compare Two Prompt Styles")

    if len(sel_styles) < 2:
        st.info("Select at least two prompt styles to enable comparison.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            style_a = st.selectbox(
                "Left prompt style",
                style_order,
                index=0,
                format_func=lambda s: STYLE_LABELS.get(s, s),
                key="style_a",
            )
        with col_right:
            default_b = 1 if len(style_order) > 1 else 0
            style_b = st.selectbox(
                "Right prompt style",
                style_order,
                index=default_b,
                format_func=lambda s: STYLE_LABELS.get(s, s),
                key="style_b",
            )

        # Sample selector
        samples_a = valid[valid["prompt_style"] == style_a].sort_values("score", ascending=False)
        samples_b = valid[valid["prompt_style"] == style_b].sort_values("score", ascending=False)

        common_samples = sorted(set(samples_a["sample_id"]) & set(samples_b["sample_id"]))

        if not common_samples:
            st.warning("No matching samples for this pair.")
        else:
            sample_mode = st.radio(
                "Sample selection",
                ["Same sample ID (paired comparison)", "Independent (pick separately)"],
                horizontal=True,
            )

            if sample_mode == "Same sample ID (paired comparison)":
                # Build labels showing scores from both sides
                pair_labels = {}
                for sid in common_samples:
                    score_a = samples_a[samples_a["sample_id"] == sid]["score"].values
                    score_b = samples_b[samples_b["sample_id"] == sid]["score"].values
                    sa = f"{score_a[0]:.0f}" if len(score_a) else "?"
                    sb = f"{score_b[0]:.0f}" if len(score_b) else "?"
                    pair_labels[sid] = f"Sample {sid}  —  {STYLE_LABELS[style_a]}: {sa}  vs  {STYLE_LABELS[style_b]}: {sb}"

                sel_sample = st.selectbox(
                    "Sample",
                    common_samples,
                    format_func=lambda s: pair_labels[s],
                )

                row_a = samples_a[samples_a["sample_id"] == sel_sample].iloc[0]
                row_b = samples_b[samples_b["sample_id"] == sel_sample].iloc[0]
            else:
                col_pick_a, col_pick_b = st.columns(2)
                with col_pick_a:
                    opts_a = []
                    for _, r in samples_a.iterrows():
                        opts_a.append(f"sample {int(r['sample_id'])} — score {r['score']:.0f}")
                    sel_a_idx = st.selectbox("Left sample", range(len(opts_a)),
                                             format_func=lambda i: opts_a[i], key="pick_a")
                    row_a = samples_a.iloc[sel_a_idx]
                with col_pick_b:
                    opts_b = []
                    for _, r in samples_b.iterrows():
                        opts_b.append(f"sample {int(r['sample_id'])} — score {r['score']:.0f}")
                    sel_b_idx = st.selectbox("Right sample", range(len(opts_b)),
                                             format_func=lambda i: opts_b[i], key="pick_b")
                    row_b = samples_b.iloc[sel_b_idx]

            # ── Render the comparison ──────────────────────────────────
            st.divider()
            col_a, col_b = st.columns(2)

            for col, row, style_key in [(col_a, row_a, style_a), (col_b, row_b, style_b)]:
                with col:
                    score_val = row["score"]
                    st.markdown(
                        f"### {STYLE_LABELS[style_key]}  —  score **{score_val:.0f}**\n"
                        f"sample {int(row['sample_id'])}"
                    )

                    st.caption(
                        f"reasoning_tokens={int(row.get('reasoning_tokens', 0)):,}  |  "
                        f"output_tokens={int(row.get('output_tokens', 0)):,}"
                    )

                    # Criteria breakdown
                    if criteria:
                        crit_vals = {c: row[c] for c in criteria if pd.notna(row.get(c))}
                        if crit_vals:
                            st.markdown("**Criteria:**")
                            cols_inner = st.columns(min(len(crit_vals), 3))
                            for i, (c, v) in enumerate(crit_vals.items()):
                                cols_inner[i % 3].metric(c.replace("_", " ").title(), f"{v:.0f}")

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

    browse_style = st.selectbox(
        "Prompt style",
        style_order,
        format_func=lambda s: STYLE_LABELS.get(s, s),
        key="browse_style",
    )
    browse_sub = valid[valid["prompt_style"] == browse_style].sort_values("score", ascending=False)
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
            crit_vals = {c: brow_row[c] for c in criteria if pd.notna(brow_row.get(c))}
            if crit_vals:
                cols_c = st.columns(min(len(crit_vals), 6))
                for i, (c, v) in enumerate(crit_vals.items()):
                    cols_c[i % 6].metric(c.replace("_", " ").title(), f"{v:.0f}")

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

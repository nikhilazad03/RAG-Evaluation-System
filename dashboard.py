import json
import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("📊 RAG Evaluation System Dashboard")
st.markdown("Comparing RAG pipeline experiments across **Answer Relevancy**, **Faithfulness**, **Contextual Relevancy**, and **Correctness**.")

# 1. Load experiment summaries
summary_files = sorted(glob.glob("experiment_v*.jsonl"))

if not summary_files:
    st.warning("No experiment summary files found matching `experiment_v*.jsonl`.")
    st.stop()

records = []
for file_path in summary_files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

df = pd.DataFrame(records)

# Format table columns
metrics_cols = ["answer_relevancy", "faithfulness", "context_relevancy", "correctness"]
display_cols = ["experiment_name", "retriever_k", "total_test_cases"] + metrics_cols

st.subheader("1. Experiment Run Overview")
st.dataframe(df[display_cols].style.highlight_max(subset=metrics_cols, color="#90EE90", axis=0), use_container_width=True)

# 2. Metric Comparison Chart
st.subheader("2. Metric Comparison by Experiment")
chart_df = df.set_index("experiment_name")[metrics_cols]
st.bar_chart(chart_df, stack=False)

# 3. Best Performing Configuration
st.subheader("3. Best Performing Run by Metric")
best_runs = []
for m in metrics_cols:
    best_row = df.loc[df[m].idxmax()]
    best_runs.append({
        "Metric": m.replace("_", " ").title(),
        "Best Experiment": best_row["experiment_name"],
        "Top Score": best_row[m],
        "Retriever K": best_row["retriever_k"],
    })
st.table(pd.DataFrame(best_runs))

# 4. Individual Run Inspector
st.subheader("4. Run Inspector")
selected_run = st.selectbox("Select Experiment to Inspect:", df["experiment_name"].tolist())
run_details = df[df["experiment_name"] == selected_run].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Answer Relevancy", f"{run_details['answer_relevancy']:.3f}")
col2.metric("Faithfulness", f"{run_details['faithfulness']:.3f}")
col3.metric("Context Relevancy", f"{run_details['context_relevancy']:.3f}")
col4.metric("Correctness", f"{run_details['correctness']:.3f}")
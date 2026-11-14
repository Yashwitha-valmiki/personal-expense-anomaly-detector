"""Streamlit dashboard for the personal expense anomaly detector."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from expense_anomaly import (
    detect_anomalies,
    evaluate_known_labels,
    generate_transactions,
    normalize_transactions,
)


st.set_page_config(page_title="Expense Anomaly Detector", page_icon="💳", layout="wide")
st.title("💳 Personal Expense Anomaly Detector")
st.caption("A privacy-first demonstration of personalized spending anomaly detection.")

with st.sidebar:
    st.header("Data and model controls")
    uploaded_file = st.file_uploader("Upload transactions CSV", type="csv")
    transaction_count = st.slider("Synthetic transactions", min_value=100, max_value=2000, value=500, step=100)
    seed = st.number_input("Synthetic data seed", min_value=0, max_value=10000, value=42)
    contamination = st.slider("Expected anomaly rate", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

try:
    if uploaded_file is not None:
        transactions = normalize_transactions(pd.read_csv(uploaded_file))
        st.sidebar.success(f"Loaded {len(transactions):,} transactions")
    else:
        transactions = generate_transactions(transaction_count, int(seed))
        st.sidebar.info("Using reproducible synthetic data")
    results, _ = detect_anomalies(transactions, contamination=contamination)
except ValueError as error:
    st.error(str(error))
    st.stop()

anomalies = results[results["is_anomaly"]]

metric_columns = st.columns(4)
metric_columns[0].metric("Transactions", len(results))
metric_columns[1].metric("Flagged", len(anomalies))
metric_columns[2].metric("Flag rate", f"{len(anomalies) / len(results):.1%}")
metric_columns[3].metric("Total spend", f"${results['amount'].sum():,.2f}")

known_metrics = evaluate_known_labels(results)
if known_metrics is not None:
    st.subheader("Synthetic-data evaluation")
    evaluation_columns = st.columns(3)
    evaluation_columns[0].metric("Precision", f"{known_metrics['precision']:.1%}")
    evaluation_columns[1].metric("Recall", f"{known_metrics['recall']:.1%}")
    evaluation_columns[2].metric("F1 score", f"{known_metrics['f1']:.1%}")

st.subheader("Spending overview")
chart_columns = st.columns(2)
with chart_columns[0]:
    st.bar_chart(results.groupby("category")["amount"].sum().sort_values(ascending=False))
with chart_columns[1]:
    st.line_chart(results.set_index("date")["amount"].resample("D").sum())

st.subheader("Anomaly distribution")
st.bar_chart(results.assign(status=results["is_anomaly"].map({True: "Flagged", False: "Normal"}))["status"].value_counts())

st.subheader("Flagged transactions")
if anomalies.empty:
    st.success("No transactions were flagged with the current settings.")
else:
    display_columns = ["transaction_id", "date", "category", "amount", "hour", "channel", "anomaly_score", "reason"]
    st.dataframe(
        anomalies[display_columns].style.format({"amount": "${:,.2f}", "anomaly_score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download analysis CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="expense_anomaly_analysis.csv",
    mime="text/csv",
)

with st.expander("How it works"):
    st.write(
        "Isolation Forest learns the user's normal combination of amount, category, time, "
        "day, and payment channel. Transactions that are easier to isolate from the normal "
        "pattern receive higher anomaly scores. This demo uses generated data and should not "
        "be treated as a fraud detector or financial advice."
    )

with st.expander("CSV format"):
    st.write("Required columns: `date`, `category`, and `amount`. Optional columns are `transaction_id`, `hour`, and `channel`.")
    st.dataframe(
        generate_transactions(3, seed=1)[["transaction_id", "date", "category", "amount", "hour", "channel"]],
        hide_index=True,
        use_container_width=True,
    )

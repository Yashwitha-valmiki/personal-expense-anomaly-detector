# Personal Expense Anomaly Detector

An educational machine-learning application that learns a user's normal spending patterns and flags unusual transactions. It creates a reproducible synthetic dataset, so no private financial data is required.

## Features

- Generates realistic synthetic personal transactions with a small number of injected anomalies.
- Uses Isolation Forest for unsupervised anomaly detection.
- Adds human-readable reasons for each flagged transaction.
- Accepts a CSV upload or uses a reproducible synthetic dataset.
- Provides a Streamlit dashboard with summary metrics, evaluation scores, charts, and a downloadable report.
- Includes validation and tests for data generation, model output, CSV input, and evaluation behavior.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

The app generates data on demand. Use the sidebar controls to change the number of transactions and random seed, or upload a CSV.

Uploaded CSV files require `date`, `category`, and `amount` columns. `transaction_id`, `hour`, and `channel` are optional; missing optional values are generated safely. Do not upload names, account numbers, card numbers, or other personally identifying information.

## Run tests

```bash
pytest -q
```

## Project structure

```text
app.py                         Streamlit dashboard
src/expense_anomaly/           Reusable data and model code
tests/                         Focused unit tests
```

This project is for learning and personal budgeting support. It is not a fraud detector or financial advice system.

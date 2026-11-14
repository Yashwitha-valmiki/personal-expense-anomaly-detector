# Personal Expense Anomaly Detector

An educational machine-learning project that learns a person's normal spending
patterns and identifies transactions that are unusually different. The project
works without a real financial dataset by generating a reproducible,
privacy-safe synthetic transaction dataset from scratch.

> **Scope:** This is a budgeting and machine-learning demonstration. It is not
> a fraud detector, banking product, credit decision system, or financial advice.

## Problem statement

People often make many transactions across different categories, payment
channels, amounts, and times of day. A transaction can be unusual for one
person even when it is normal for another. The goal of this project is to build
an unsupervised machine-learning system that learns an individual's regular
spending behavior and flags transactions that deviate from that pattern.

The system produces:

- A normal or anomalous label for each transaction
- An anomaly score for ranking unusual transactions
- A plain-language explanation for each flagged transaction
- A CSV report that can be reviewed or visualized

## Project objectives

1. Generate a realistic, reproducible transaction dataset without exposing
   private financial information.
2. Preprocess numerical and categorical transaction features.
3. Detect unusual spending without requiring manually labeled fraud examples.
4. Explain why a transaction was flagged.
5. Provide a repeatable command-line workflow suitable for an academic
   mini-project.

## Machine-learning approach

### Algorithm: Isolation Forest

The main model is **Isolation Forest**, an unsupervised anomaly-detection
algorithm. It repeatedly partitions the feature space. Transactions that are
isolated in fewer partitions are considered more unusual and receive higher
anomaly scores.

Isolation Forest is a good fit because:

- Real anomaly labels are usually unavailable.
- It works with small and medium-sized tabular datasets.
- It can model combinations of amount, category, time, and channel.
- It is more appropriate for this problem than a standard supervised
  classifier when labels are not available.

### Features used

- `amount`: transaction value
- `category`: spending category
- `hour`: hour of the transaction
- `day_of_week`: derived from the transaction date
- `channel`: cash, card, online, or another channel

Categorical fields are one-hot encoded and numerical fields are standardized
inside a scikit-learn pipeline.

### Explanations

The model's score is supplemented with transparent checks against the learned
transaction data. A flagged transaction may be explained by:

- An amount much higher than the usual amount for its category
- An unusual transaction hour
- A different payment channel
- A combination of features that differs from the normal pattern

These explanations improve interpretability, but they should not be treated as
proof that a transaction is fraudulent.

## Dataset strategy

No external or private dataset is required. `generate_transactions()` creates
synthetic transactions using category-specific spending profiles and injects a
small number of intentionally unusual records. The fixed random seed makes the
dataset reproducible.

The generated records contain:

```text
transaction_id, date, category, amount, hour, channel, injected_anomaly
```

The `injected_anomaly` column is only used for evaluating the synthetic demo.
It is not used as a model feature.

The code also supports CSV input. A user CSV requires:

```text
date, category, amount
```

Optional columns are `transaction_id`, `hour`, and `channel`. Do not upload
names, account numbers, card numbers, or other personally identifying data.

## Repository structure

```text
personal-expense-anomaly-detector/
├── run_analysis.py                    # Main assignment command
├── app.py                             # Optional Streamlit dashboard
├── requirements.txt                   # Core dependencies
├── requirements-dashboard.txt         # Optional dashboard dependency
├── src/
│   └── expense_anomaly/
│       ├── __init__.py
│       ├── data.py                    # Synthetic data and CSV validation
│       └── model.py                   # Isolation Forest and evaluation
├── tests/
│   └── test_project.py                # Automated tests
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the assignment pipeline

From the repository root:

```bash
python run_analysis.py
```

The command:

1. Generates 500 reproducible synthetic transactions.
2. Trains the Isolation Forest model.
3. Flags unusual transactions.
4. Prints a summary.
5. Writes the result to:

```text
outputs/expense_anomaly_analysis.csv
```

### Command options

```bash
python run_analysis.py --transactions 1000 --seed 7 --contamination 0.05
```

Available options:

| Option | Default | Description |
|---|---:|---|
| `--transactions` | `500` | Number of synthetic transactions |
| `--seed` | `42` | Random seed |
| `--contamination` | `0.05` | Expected fraction of anomalies |
| `--output` | `outputs/expense_anomaly_analysis.csv` | Output CSV path |

## Run the tests

```bash
pytest -q
```

The tests cover:

- Reproducible data generation
- Anomaly detection output
- Input validation
- Minimal CSV normalization
- Invalid amount rejection
- Evaluation metrics for synthetic labels

## Optional dashboard

The dashboard is not required for the assignment pipeline. It is included as
an optional way to demonstrate the project visually.

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

The dashboard provides summary metrics, spending charts, flagged transactions,
explanations, CSV upload, and report download.

## Evaluation

For generated data, the project compares predictions with the hidden
`injected_anomaly` flag and reports:

- **Precision:** proportion of flagged records that were injected anomalies
- **Recall:** proportion of injected anomalies that were detected
- **F1 score:** balance between precision and recall

These scores demonstrate behavior on the synthetic dataset only. They do not
represent performance on real financial data.

## Limitations and ethical considerations

- Synthetic data cannot represent every real spending pattern.
- An anomaly is not necessarily fraud or an incorrect transaction.
- A user may have legitimate unusual expenses.
- The model should not automatically block payments or accuse users.
- Real financial data must be anonymized and handled securely.
- Model performance should be tested separately before any real-world use.

## Future improvements

- Add a user-specific history and rolling spending baselines.
- Compare Isolation Forest with Local Outlier Factor.
- Add a configurable currency and locale.
- Add time-series features such as weekly and monthly spending trends.
- Add model monitoring for changing spending behavior.
- Evaluate on a carefully anonymized real-world dataset.

## License

This project is intended for educational use. Add a repository license before
redistributing or using it in a larger project.

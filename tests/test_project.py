import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from expense_anomaly import (
    detect_anomalies,
    evaluate_known_labels,
    generate_transactions,
    normalize_transactions,
)


def test_generator_is_reproducible_and_injects_anomalies():
    first = generate_transactions(100, seed=7)
    second = generate_transactions(100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["injected_anomaly"].sum() > 0


def test_detector_returns_scores_flags_and_reasons():
    transactions = generate_transactions(150, seed=3)
    result, pipeline = detect_anomalies(transactions)
    assert len(result) == 150
    assert {"anomaly_score", "is_anomaly", "reason"}.issubset(result.columns)
    assert result["is_anomaly"].any()
    assert pipeline.named_steps["model"].n_estimators == 250


def test_invalid_transaction_count_is_rejected():
    with pytest.raises(ValueError, match="at least 20"):
        generate_transactions(10)


def test_normalize_transactions_supports_minimal_csv_shape():
    raw = pd.DataFrame(
        {
            "date": ["2026-01-01 09:30"],
            "category": ["Groceries"],
            "amount": ["25.50"],
        }
    )
    normalized = normalize_transactions(raw)
    assert normalized.loc[0, "hour"] == 9
    assert normalized.loc[0, "channel"] == "Unknown"


def test_normalize_transactions_rejects_invalid_amount():
    raw = pd.DataFrame({"date": ["2026-01-01"], "category": ["Dining"], "amount": [-2]})
    with pytest.raises(ValueError, match="positive"):
        normalize_transactions(raw)


def test_known_labels_produce_evaluation_metrics():
    results, _ = detect_anomalies(generate_transactions(200, seed=9))
    metrics = evaluate_known_labels(results)
    assert metrics is not None
    assert set(metrics) == {"precision", "recall", "f1"}

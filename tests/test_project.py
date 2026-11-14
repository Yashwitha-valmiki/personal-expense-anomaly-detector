import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from expense_anomaly import detect_anomalies, generate_transactions


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

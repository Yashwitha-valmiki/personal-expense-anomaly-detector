"""Isolation Forest model and explanation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_pipeline(contamination: float, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["amount", "hour", "day_of_week"]),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category", "channel"]),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _reason(row: pd.Series, category_median: pd.Series) -> str:
    reasons: list[str] = []
    median = category_median.get(row["category"], row["amount"])
    if row["amount"] > median * 2.5:
        reasons.append("amount is much higher than usual for this category")
    if row["hour"] < 6 or row["hour"] >= 23:
        reasons.append("transaction occurred at an unusual hour")
    if row["channel"] == "Online":
        reasons.append("online purchase differs from the common spending channel")
    return "; ".join(reasons) or "combination of spending features differs from the learned pattern"


def detect_anomalies(
    transactions: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Pipeline]:
    """Fit Isolation Forest and return transactions with scores and explanations."""
    required = {"transaction_id", "category", "amount", "hour", "channel"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    result = transactions.copy()
    result["day_of_week"] = pd.to_datetime(result["date"]).dt.dayofweek if "date" in result else 0
    pipeline = _build_pipeline(contamination, random_state)
    features = result[["amount", "hour", "day_of_week", "category", "channel"]]
    pipeline.fit(features)
    result["anomaly_score"] = -pipeline.decision_function(features)
    result["is_anomaly"] = pipeline.predict(features) == -1
    category_median = result.groupby("category")["amount"].median()
    result["reason"] = result.apply(
        lambda row: _reason(row, category_median) if row["is_anomaly"] else "within the learned personal spending pattern",
        axis=1,
    )
    return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True), pipeline

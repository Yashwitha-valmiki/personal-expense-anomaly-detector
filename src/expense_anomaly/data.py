"""Synthetic transaction data generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CategoryProfile:
    """Typical spending profile for one transaction category."""

    amount_mean: float
    amount_std: float
    daily_probability: float


CATEGORY_PROFILES: dict[str, CategoryProfile] = {
    "Groceries": CategoryProfile(48, 18, 0.18),
    "Transport": CategoryProfile(14, 7, 0.24),
    "Dining": CategoryProfile(32, 16, 0.16),
    "Entertainment": CategoryProfile(45, 22, 0.07),
    "Utilities": CategoryProfile(95, 25, 0.04),
    "Shopping": CategoryProfile(85, 45, 0.08),
    "Health": CategoryProfile(55, 30, 0.04),
    "Subscriptions": CategoryProfile(18, 5, 0.03),
}

REQUIRED_COLUMNS = {"transaction_id", "date", "category", "amount", "channel"}


def normalize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a transaction table for model inference."""
    result = transactions.copy()
    if "transaction_id" not in result.columns:
        result.insert(0, "transaction_id", [f"TXN-{index + 1:05d}" for index in range(len(result))])
    if "channel" not in result.columns:
        result["channel"] = "Unknown"
    if "hour" not in result.columns and "date" in result.columns:
        parsed_dates = pd.to_datetime(result["date"], errors="coerce")
        result["hour"] = parsed_dates.dt.hour

    missing = REQUIRED_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    result["hour"] = pd.to_numeric(result["hour"], errors="coerce")
    if result[["date", "amount", "hour"]].isna().any().any():
        raise ValueError("date, amount, and hour must contain valid values")
    if (result["amount"] <= 0).any() or ~result["amount"].map(np.isfinite).all():
        raise ValueError("amount must contain only positive finite values")
    if not result["hour"].between(0, 23).all():
        raise ValueError("hour must be between 0 and 23")
    if result.empty:
        raise ValueError("at least one transaction is required")
    return result.sort_values("date").reset_index(drop=True)


def generate_transactions(
    n_transactions: int = 500,
    seed: int = 42,
    anomaly_fraction: float = 0.04,
) -> pd.DataFrame:
    """Generate reproducible synthetic transactions with injected anomalies."""
    if n_transactions < 20:
        raise ValueError("n_transactions must be at least 20")
    if not 0 <= anomaly_fraction < 0.5:
        raise ValueError("anomaly_fraction must be between 0 and 0.5")

    rng = np.random.default_rng(seed)
    categories = list(CATEGORY_PROFILES)
    weights = np.array([profile.daily_probability for profile in CATEGORY_PROFILES.values()])
    weights = weights / weights.sum()
    category_values = rng.choice(categories, size=n_transactions, p=weights)
    dates = pd.date_range(end=pd.Timestamp("2026-01-01"), periods=180)
    transaction_dates = rng.choice(dates, size=n_transactions)

    amounts = np.empty(n_transactions)
    hours = rng.choice(np.arange(7, 23), size=n_transactions)
    channels = rng.choice(["Card", "Cash", "Online"], size=n_transactions, p=[0.58, 0.16, 0.26])

    for index, category in enumerate(category_values):
        profile = CATEGORY_PROFILES[category]
        amounts[index] = max(1.0, rng.normal(profile.amount_mean, profile.amount_std))

    anomaly_count = max(1, int(n_transactions * anomaly_fraction))
    anomaly_indexes = rng.choice(n_transactions, size=anomaly_count, replace=False)
    for index in anomaly_indexes:
        category = category_values[index]
        amounts[index] = CATEGORY_PROFILES[category].amount_mean * rng.uniform(4.0, 9.0)
        hours[index] = int(rng.choice([0, 1, 2, 3, 4, 5]))
        channels[index] = "Online"

    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{index + 1:05d}" for index in range(n_transactions)],
            "date": pd.to_datetime(transaction_dates),
            "category": category_values,
            "amount": np.round(amounts, 2),
            "hour": hours,
            "channel": channels,
            "injected_anomaly": False,
        }
    )
    transactions.loc[anomaly_indexes, "injected_anomaly"] = True
    return normalize_transactions(transactions)

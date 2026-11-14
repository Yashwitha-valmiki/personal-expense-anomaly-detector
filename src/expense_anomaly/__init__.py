"""Personal expense anomaly detection package."""

from .data import generate_transactions
from .model import detect_anomalies

__all__ = ["detect_anomalies", "generate_transactions"]

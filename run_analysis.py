"""Run the anomaly detector without a web application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from expense_anomaly import detect_anomalies, generate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transactions and detect unusual expenses.")
    parser.add_argument("--transactions", type=int, default=500, help="Number of synthetic transactions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/expense_anomaly_analysis.csv"),
        help="Path for the analysis CSV.",
    )
    args = parser.parse_args()

    transactions = generate_transactions(args.transactions, args.seed)
    results, _ = detect_anomalies(transactions, contamination=args.contamination)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    flagged = int(results["is_anomaly"].sum())
    print(f"Analyzed {len(results):,} transactions.")
    print(f"Flagged {flagged:,} unusual transactions ({flagged / len(results):.1%}).")
    print(f"Saved report to {args.output}.")


if __name__ == "__main__":
    main()

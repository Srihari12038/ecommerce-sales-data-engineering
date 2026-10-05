from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "order_anomaly_dataset.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "predictions"
    / "order_anomaly_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "order_anomaly_analysis.csv"
)


# ---------------------------------------------------------------------
# Business features to inspect
# ---------------------------------------------------------------------
ANALYSIS_FEATURES = [
    "item_count",
    "unique_products",
    "unique_sellers",
    "total_price",
    "total_freight",
    "total_order_value",
    "average_price",
    "average_freight",
    "payment_count",
    "total_payment_value",
    "max_installments",
    "average_installments",
    "freight_ratio",
    "payment_to_order_ratio",
    "average_item_value",
]


# ---------------------------------------------------------------------
# Percentile-based unusual-value thresholds
# ---------------------------------------------------------------------
HIGH_PERCENTILE = 0.99
LOW_PERCENTILE = 0.01


def describe_anomaly_features(row, thresholds):
    reasons = []

    # High-side anomalies.
    high_rules = [
        ("item_count", "Very high item count"),
        ("unique_products", "Many unique products"),
        ("unique_sellers", "Many unique sellers"),
        ("total_price", "Very high product value"),
        ("total_freight", "Very high freight"),
        ("total_order_value", "Very high order value"),
        ("average_price", "Very high average item price"),
        ("average_freight", "Very high average freight"),
        ("payment_count", "Unusually many payments"),
        ("total_payment_value", "Very high payment value"),
        ("max_installments", "High installment count"),
        ("average_installments", "High average installments"),
        ("freight_ratio", "Extreme freight ratio"),
        ("average_item_value", "Very high average item value"),
    ]

    for feature, description in high_rules:
        threshold = thresholds[feature]["high"]

        if row[feature] >= threshold:
            reasons.append(description)

    # Low-side unusual payment ratio.
    payment_ratio = row["payment_to_order_ratio"]

    if payment_ratio <= thresholds[
        "payment_to_order_ratio"
    ]["low"]:
        reasons.append(
            "Unusually low payment-to-order ratio"
        )

    if payment_ratio >= thresholds[
        "payment_to_order_ratio"
    ]["high"]:
        reasons.append(
            "Unusually high payment-to-order ratio"
        )

    if not reasons:
        reasons.append(
            "Multivariate Isolation Forest pattern"
        )

    return " | ".join(reasons)


def main():
    print("=" * 70)
    print("ANALYZING ORDER-LEVEL ANOMALIES")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate files
    # -----------------------------------------------------------------
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------
    data = pd.read_csv(
        DATASET_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    # -----------------------------------------------------------------
    # Merge business features with anomaly results
    # -----------------------------------------------------------------
    df = predictions.merge(
        data,
        on=[
            "order_id",
            "order_purchase_timestamp",
            "order_status",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(data):
        raise ValueError(
            "Prediction merge did not preserve all orders."
        )

    # -----------------------------------------------------------------
    # Percentile thresholds calculated from ALL orders.
    # -----------------------------------------------------------------
    thresholds = {}

    for feature in ANALYSIS_FEATURES:
        thresholds[feature] = {
            "low": data[feature].quantile(
                LOW_PERCENTILE
            ),
            "high": data[feature].quantile(
                HIGH_PERCENTILE
            ),
        }

    # -----------------------------------------------------------------
    # Add business interpretation.
    # -----------------------------------------------------------------
    anomalous = df[
        df["anomaly_flag"] == 1
    ].copy()

    anomalous["business_reason"] = anomalous.apply(
        lambda row: describe_anomaly_features(
            row,
            thresholds,
        ),
        axis=1,
    )

    # -----------------------------------------------------------------
    # Compare anomalous vs normal groups.
    # -----------------------------------------------------------------
    normal = df[
        df["anomaly_flag"] == 0
    ]

    comparison_rows = []

    for feature in ANALYSIS_FEATURES:
        comparison_rows.append(
            {
                "feature": feature,
                "normal_mean": normal[feature].mean(),
                "anomaly_mean": anomalous[feature].mean(),
                "normal_median": normal[feature].median(),
                "anomaly_median": anomalous[feature].median(),
                "normal_max": normal[feature].max(),
                "anomaly_max": anomalous[feature].max(),
                "p99_all_orders": data[
                    feature
                ].quantile(0.99),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # -----------------------------------------------------------------
    # Top 20 anomalies
    # -----------------------------------------------------------------
    top_anomalies = (
        anomalous
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    # -----------------------------------------------------------------
    # Print top anomalies with business features.
    # -----------------------------------------------------------------
    print("\nTOP 20 ANOMALOUS ORDERS")
    print("-" * 70)

    columns_to_print = [
        "anomaly_rank",
        "order_id",
        "anomaly_score",
        "business_reason",
        "item_count",
        "unique_products",
        "unique_sellers",
        "total_order_value",
        "total_freight",
        "payment_count",
        "max_installments",
        "freight_ratio",
    ]

    print(
        top_anomalies[
            columns_to_print
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Summary statistics.
    # -----------------------------------------------------------------
    print("\nANOMALY VS NORMAL")
    print("-" * 70)

    print(
        f"Normal orders:     {len(normal):,}"
    )

    print(
        f"Anomalous orders:  {len(anomalous):,}"
    )

    print(
        f"Anomaly rate:      "
        f"{len(anomalous) / len(df) * 100:.2f}%"
    )

    # -----------------------------------------------------------------
    # Average comparisons.
    # -----------------------------------------------------------------
    print("\nFEATURE COMPARISON")
    print("-" * 70)

    for feature in ANALYSIS_FEATURES:
        normal_mean = normal[
            feature
        ].mean()

        anomaly_mean = anomalous[
            feature
        ].mean()

        print(
            f"{feature:<25}"
            f" normal={normal_mean:.4f}"
            f" anomaly={anomaly_mean:.4f}"
        )

    # -----------------------------------------------------------------
    # Business-reason distribution.
    # -----------------------------------------------------------------
    print("\nBUSINESS REASON DISTRIBUTION")
    print("-" * 70)

    reason_counts = (
        anomalous["business_reason"]
        .value_counts()
        .head(20)
    )

    print(
        reason_counts.to_string()
    )

    # -----------------------------------------------------------------
    # Save detailed anomaly analysis.
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "anomaly_rank",
        "order_id",
        "order_purchase_timestamp",
        "order_status",
        "anomaly_score",
        "anomaly_flag",
        "anomaly_label",
        "business_reason",
    ] + ANALYSIS_FEATURES

    anomalous[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Also save group comparison.
    # -----------------------------------------------------------------
    comparison_path = (
        OUTPUT_DIR
        / "order_anomaly_feature_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)
    print(comparison_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

PREDICTION_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "predictions"
    / "order_anomaly_predictions.csv"
)

ANALYSIS_PATH = (
    EVALUATION_DIR
    / "order_anomaly_analysis.csv"
)

FEATURE_COMPARISON_PATH = (
    EVALUATION_DIR
    / "order_anomaly_feature_comparison.csv"
)

SENSITIVITY_PATH = (
    EVALUATION_DIR
    / "order_anomaly_contamination_sensitivity.csv"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "order_anomaly_report.csv"
)


SELECTED_CONTAMINATION = 0.01


def main():
    print("=" * 70)
    print("CREATING ORDER ANOMALY REPORT")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate required files
    # -----------------------------------------------------------------
    required_files = [
        PREDICTION_PATH,
        ANALYSIS_PATH,
        FEATURE_COMPARISON_PATH,
        SENSITIVITY_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # -----------------------------------------------------------------
    # Load outputs
    # -----------------------------------------------------------------
    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    analysis = pd.read_csv(
        ANALYSIS_PATH
    )

    feature_comparison = pd.read_csv(
        FEATURE_COMPARISON_PATH
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_PATH
    )

    # -----------------------------------------------------------------
    # Selected contamination row
    # -----------------------------------------------------------------
    selected_sensitivity = sensitivity[
        sensitivity["contamination"]
        == SELECTED_CONTAMINATION
    ]

    if selected_sensitivity.empty:
        raise ValueError(
            "Selected contamination value was not found "
            "in the sensitivity results."
        )

    selected_sensitivity = (
        selected_sensitivity.iloc[0]
    )

    # -----------------------------------------------------------------
    # Overall anomaly statistics
    # -----------------------------------------------------------------
    total_orders = len(predictions)

    anomalous_orders = int(
        predictions[
            "anomaly_flag"
        ].sum()
    )

    normal_orders = (
        total_orders
        - anomalous_orders
    )

    anomaly_rate = (
        anomalous_orders
        / total_orders
    )

    # -----------------------------------------------------------------
    # Top anomaly
    # -----------------------------------------------------------------
    top_anomaly = (
        predictions
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .iloc[0]
    )

    # -----------------------------------------------------------------
    # Mean anomaly score
    # -----------------------------------------------------------------
    mean_anomaly_score = (
        predictions[
            "anomaly_score"
        ].mean()
    )

    median_anomaly_score = (
        predictions[
            "anomaly_score"
        ].median()
    )

    # -----------------------------------------------------------------
    # Most common business reasons
    # -----------------------------------------------------------------
    if "business_reason" in analysis.columns:
        top_reasons = (
            analysis[
                "business_reason"
            ]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_reasons.columns = [
            "business_reason",
            "count",
        ]

    else:
        top_reasons = pd.DataFrame(
            columns=[
                "business_reason",
                "count",
            ]
        )

    # -----------------------------------------------------------------
    # Feature comparison summary
    # -----------------------------------------------------------------
    strongest_feature = None

    if not feature_comparison.empty:
        feature_comparison = feature_comparison.copy()

        feature_comparison[
            "difference"
        ] = (
            feature_comparison[
                "anomaly_mean"
            ]
            - feature_comparison[
                "normal_mean"
            ]
        )

        strongest_feature = (
            feature_comparison
            .sort_values(
                "difference",
                ascending=False,
            )
            .iloc[0]["feature"]
        )

    # -----------------------------------------------------------------
    # Build final report
    # -----------------------------------------------------------------
    report = pd.DataFrame(
        [
            {
                "module": "Order-Level Anomaly Detection",
                "model": "Isolation Forest",
                "n_estimators": 300,
                "selected_contamination": (
                    SELECTED_CONTAMINATION
                ),
                "total_orders": total_orders,
                "normal_orders": normal_orders,
                "anomalous_orders": anomalous_orders,
                "anomaly_rate": anomaly_rate,
                "mean_anomaly_score": mean_anomaly_score,
                "median_anomaly_score": median_anomaly_score,
                "top_anomaly_order_id": (
                    top_anomaly["order_id"]
                ),
                "top_anomaly_score": (
                    top_anomaly["anomaly_score"]
                ),
                "sensitivity_0_5_anomalies": int(
                    sensitivity.loc[
                        sensitivity["contamination"]
                        == 0.005,
                        "anomalous_orders",
                    ].iloc[0]
                ),
                "sensitivity_1_0_anomalies": int(
                    selected_sensitivity[
                        "anomalous_orders"
                    ]
                ),
                "sensitivity_2_0_anomalies": int(
                    sensitivity.loc[
                        sensitivity["contamination"]
                        == 0.020,
                        "anomalous_orders",
                    ].iloc[0]
                ),
                "strongest_business_feature": (
                    strongest_feature
                ),
                "model_status": (
                    "Selected unsupervised anomaly detector"
                ),
                "business_interpretation": (
                    "The Isolation Forest identifies a small "
                    "population of unusually large or structurally "
                    "complex transactions. These are transactional "
                    "anomalies and are not confirmed fraud cases."
                ),
                "contamination_interpretation": (
                    "The 1% contamination setting is an operating "
                    "assumption selected because the anomalous "
                    "population remains materially different from "
                    "normal orders across tested contamination levels."
                ),
            }
        ]
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------
    print("\nFINAL ANOMALY MODEL")
    print("-" * 70)

    print(
        "Model: Isolation Forest"
    )

    print(
        f"Selected contamination: "
        f"{SELECTED_CONTAMINATION:.2%}"
    )

    print(
        f"Total orders:      "
        f"{total_orders:,}"
    )

    print(
        f"Normal orders:     "
        f"{normal_orders:,}"
    )

    print(
        f"Anomalous orders:  "
        f"{anomalous_orders:,}"
    )

    print(
        f"Anomaly rate:      "
        f"{anomaly_rate * 100:.2f}%"
    )

    print("\nSENSITIVITY")
    print("-" * 70)

    for _, row in sensitivity.iterrows():
        print(
            f"{row['contamination']:.2%} -> "
            f"{int(row['anomalous_orders']):,} anomalies"
        )

    print("\nBUSINESS INTERPRETATION")
    print("-" * 70)

    print(
        "The detector primarily identifies unusually large "
        "or structurally complex transactions."
    )

    print(
        "These are transactional anomalies, not confirmed fraud."
    )

    print(
        "\nSaved:"
    )
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
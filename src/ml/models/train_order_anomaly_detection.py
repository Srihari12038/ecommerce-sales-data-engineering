from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "order_anomaly_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "src"
    / "ml"
    / "models"
)

PREDICTION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "predictions"
)

EVALUATION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

MODEL_PATH = (
    MODEL_DIR
    / "order_isolation_forest.pkl"
)

PREDICTION_PATH = (
    PREDICTION_DIR
    / "order_anomaly_predictions.csv"
)

SUMMARY_PATH = (
    EVALUATION_DIR
    / "order_anomaly_summary.csv"
)


# ---------------------------------------------------------------------
# Initial contamination assumption.
# This will be tested later through sensitivity analysis.
# ---------------------------------------------------------------------
CONTAMINATION = 0.01


def main():
    print("=" * 70)
    print("ORDER-LEVEL ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input feature file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    # -----------------------------------------------------------------
    # Select only scaled model features.
    # -----------------------------------------------------------------
    feature_columns = [
        column
        for column in df.columns
        if column.startswith("scaled_")
    ]

    if not feature_columns:
        raise ValueError(
            "No scaled anomaly features were found."
        )

    X = df[
        feature_columns
    ].copy()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    missing_values = int(
        X.isna().sum().sum()
    )

    infinite_values = int(
        np.isinf(
            X.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Model features contain {missing_values} missing values."
        )

    if infinite_values > 0:
        raise ValueError(
            f"Model features contain {infinite_values} infinite values."
        )

    # -----------------------------------------------------------------
    # Train Isolation Forest
    # -----------------------------------------------------------------
    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("\nMODEL CONFIGURATION")
    print("-" * 70)
    print(f"Orders:        {len(df):,}")
    print(f"Features:      {len(feature_columns)}")
    print(f"Estimators:    300")
    print(f"Contamination: {CONTAMINATION:.2%}")

    print("\nTraining Isolation Forest...")

    model.fit(
        X
    )

    # -----------------------------------------------------------------
    # Predictions
    #
    # IsolationForest.predict():
    #   +1 = normal
    #   -1 = anomaly
    #
    # decision_function:
    #   higher = more normal
    #   lower  = more anomalous
    #
    # anomaly_score is reversed so that:
    #   higher = more anomalous
    # -----------------------------------------------------------------
    raw_prediction = model.predict(
        X
    )

    decision_scores = model.decision_function(
        X
    )

    anomaly_scores = -decision_scores

    anomaly_flag = (
        raw_prediction == -1
    ).astype(int)

    # -----------------------------------------------------------------
    # Create prediction output
    # -----------------------------------------------------------------
    predictions = df[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_status",
        ]
    ].copy()

    predictions["anomaly_score"] = (
        anomaly_scores
    )

    predictions["anomaly_flag"] = (
        anomaly_flag
    )

    predictions["anomaly_label"] = np.where(
        anomaly_flag == 1,
        "Anomalous",
        "Normal",
    )

    # Higher score = more anomalous.
    predictions = (
        predictions
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    predictions["anomaly_rank"] = (
        np.arange(len(predictions)) + 1
    )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------
    anomaly_count = int(
        anomaly_flag.sum()
    )

    normal_count = int(
        (anomaly_flag == 0).sum()
    )

    anomaly_rate = (
        anomaly_count / len(df)
    )

    print("\nANOMALY SUMMARY")
    print("-" * 70)

    print(
        f"Normal orders:       "
        f"{normal_count:,}"
    )

    print(
        f"Anomalous orders:    "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly rate:        "
        f"{anomaly_rate * 100:.2f}%"
    )

    print(
        f"Highest anomaly score:"
        f" {anomaly_scores.max():.6f}"
    )

    print(
        f"Lowest anomaly score: "
        f"{anomaly_scores.min():.6f}"
    )

    # -----------------------------------------------------------------
    # Top anomalous orders
    # -----------------------------------------------------------------
    print("\nTOP 20 ANOMALOUS ORDERS")
    print("-" * 70)

    print(
        predictions[
            [
                "anomaly_rank",
                "order_id",
                "anomaly_score",
                "anomaly_label",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # -----------------------------------------------------------------
    # Quantiles for anomaly score
    # -----------------------------------------------------------------
    print("\nANOMALY SCORE QUANTILES")
    print("-" * 70)

    quantiles = pd.Series(
        anomaly_scores
    ).quantile(
        [
            0,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            1,
        ]
    )

    print(
        quantiles.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------------------
    # Summary output
    # -----------------------------------------------------------------
    summary = pd.DataFrame(
        [
            {
                "model": "IsolationForest",
                "n_estimators": 300,
                "contamination": CONTAMINATION,
                "total_orders": len(df),
                "normal_orders": normal_count,
                "anomalous_orders": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "mean_anomaly_score": anomaly_scores.mean(),
                "median_anomaly_score": np.median(
                    anomaly_scores
                ),
                "max_anomaly_score": anomaly_scores.max(),
            }
        ]
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    predictions.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(MODEL_PATH)
    print(PREDICTION_PATH)
    print(SUMMARY_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
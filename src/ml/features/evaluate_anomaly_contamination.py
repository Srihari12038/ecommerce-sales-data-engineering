from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "order_anomaly_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "order_anomaly_contamination_sensitivity.csv"
)


CONTAMINATION_LEVELS = [
    0.005,
    0.010,
    0.020,
]


def main():
    print("=" * 70)
    print("ORDER ANOMALY CONTAMINATION SENSITIVITY")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("scaled_")
    ]

    X = df[feature_columns]

    results = []

    for contamination in CONTAMINATION_LEVELS:

        print(
            f"\nTesting contamination: "
            f"{contamination:.1%}"
        )

        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X)

        raw_prediction = model.predict(X)

        anomaly_flags = (
            raw_prediction == -1
        ).astype(int)

        anomaly_count = int(
            anomaly_flags.sum()
        )

        anomaly_mask = (
            anomaly_flags == 1
        )

        normal_mask = (
            anomaly_flags == 0
        )

        # -------------------------------------------------------------
        # Business feature comparisons
        # -------------------------------------------------------------
        total_order_value_anomaly = (
            df.loc[
                anomaly_mask,
                "log1p_total_order_value",
            ]
            .mean()
        )

        total_order_value_normal = (
            df.loc[
                normal_mask,
                "log1p_total_order_value",
            ]
            .mean()
        )

        freight_ratio_anomaly = (
            df.loc[
                anomaly_mask,
                "freight_ratio",
            ]
            .mean()
        )

        freight_ratio_normal = (
            df.loc[
                normal_mask,
                "freight_ratio",
            ]
            .mean()
        )

        item_count_anomaly = (
            df.loc[
                anomaly_mask,
                "log1p_item_count",
            ]
            .mean()
        )

        item_count_normal = (
            df.loc[
                normal_mask,
                "log1p_item_count",
            ]
            .mean()
        )

        results.append(
            {
                "contamination": contamination,
                "total_orders": len(df),
                "anomalous_orders": anomaly_count,
                "anomaly_rate": (
                    anomaly_count / len(df)
                ),
                "log_order_value_anomaly_mean": (
                    total_order_value_anomaly
                ),
                "log_order_value_normal_mean": (
                    total_order_value_normal
                ),
                "freight_ratio_anomaly_mean": (
                    freight_ratio_anomaly
                ),
                "freight_ratio_normal_mean": (
                    freight_ratio_normal
                ),
                "log_item_count_anomaly_mean": (
                    item_count_anomaly
                ),
                "log_item_count_normal_mean": (
                    item_count_normal
                ),
            }
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRESULTS")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nINTERPRETATION")
    print("-" * 70)
    print(
        "Higher contamination produces more flagged orders."
    )
    print(
        "Compare whether the anomalous population remains "
        "materially different from the normal population."
    )
    print(
        "The contamination value should be selected based "
        "on business interpretability and stability, not "
        "treated as a ground-truth anomaly rate."
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
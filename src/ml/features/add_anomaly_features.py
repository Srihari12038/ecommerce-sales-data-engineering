import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)

ANOMALY_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "sales_anomalies.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features_anomaly_aware.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Load feature dataset
    # ---------------------------------------------------------

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"]
    )

    # ---------------------------------------------------------
    # 2. Load anomaly results
    # ---------------------------------------------------------

    anomalies = pd.read_csv(
        ANOMALY_FILE,
        parse_dates=["date"]
    )

    anomalies = anomalies[
    [
        "date",
        "sales_change_pct",
        "is_anomaly",
        "anomaly_type"
    ]
    ]

    # ---------------------------------------------------------
    # 3. Merge anomaly information
    # ---------------------------------------------------------

    df = features.merge(
        anomalies,
        on="date",
        how="left"
    )

    # ---------------------------------------------------------
    # 4. Handle missing anomaly values
    # ---------------------------------------------------------

    df["is_anomaly"] = (
        df["is_anomaly"]
        .fillna(False)
        .astype(bool)
    )

    df["anomaly_type"] = (
        df["anomaly_type"]
        .fillna("Normal")
    )

    # ---------------------------------------------------------
    # 5. Numeric anomaly indicator
    # ---------------------------------------------------------

    df["anomaly_flag"] = (
        df["is_anomaly"]
        .astype(int)
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("ANOMALY-AWARE FEATURE DATASET")
    print("=" * 80)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nAnomaly Distribution")
    print("-" * 80)

    print(
        df["anomaly_type"]
        .value_counts()
        .to_string()
    )

    print("\nDetected Anomaly Rows")
    print("-" * 80)

    print(
        df[
            df["is_anomaly"]
        ][
            [
                "date",
                "total_sales",
                "sales_change_pct",
                "anomaly_type",
                "anomaly_flag"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
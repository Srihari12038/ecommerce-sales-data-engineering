from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / "ml" / "delivery_prediction_dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "ml"


def main():
    print("=" * 70)
    print("CREATING DELIVERY ML FEATURES")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "purchase_timestamp",
            "delivered_date",
            "estimated_delivery_date",
        ],
    )

    # ------------------------------------------------------------------
    # Sort chronologically.
    # ------------------------------------------------------------------
    df = df.sort_values(
        "purchase_timestamp"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Features that are available at order time.
    # ------------------------------------------------------------------
    numeric_features = [
        "item_count",
        "unique_products",
        "unique_sellers",
        "total_price",
        "total_freight",
        "total_order_value",
        "average_price",
        "average_freight",
        "average_product_weight_g",
        "max_product_weight_g",
        "average_product_volume_cm3",
        "average_product_photos",
        "same_state_ratio",
        "estimated_delivery_days",
        "payment_count",
        "total_payment_value",
        "max_installments",
        "purchase_year",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "freight_ratio",
    ]

    categorical_features = [
        "customer_state",
    ]

    # ------------------------------------------------------------------
    # Targets.
    # ------------------------------------------------------------------
    target_regression = "delivery_days"
    target_classification = "late_delivery_flag"

    required_columns = (
        numeric_features
        + categorical_features
        + [
            target_regression,
            target_classification,
            "order_id",
            "purchase_timestamp",
        ]
    )

    df = df[required_columns].copy()

    # ------------------------------------------------------------------
    # Time-aware 80/20 split.
    # ------------------------------------------------------------------
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # ------------------------------------------------------------------
    # One-hot encode customer state using TRAINING categories only.
    # ------------------------------------------------------------------
    train_states = pd.get_dummies(
        train_df[categorical_features],
        columns=categorical_features,
        dtype=int,
    )

    test_states = pd.get_dummies(
        test_df[categorical_features],
        columns=categorical_features,
        dtype=int,
    )

    # Force test to use exactly the training feature space.
    test_states = test_states.reindex(
        columns=train_states.columns,
        fill_value=0,
    )

    train_features = pd.concat(
        [
            train_df[numeric_features].reset_index(drop=True),
            train_states.reset_index(drop=True),
        ],
        axis=1,
    )

    test_features = pd.concat(
        [
            test_df[numeric_features].reset_index(drop=True),
            test_states.reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------------------
    # Add targets.
    # ------------------------------------------------------------------
    train_features["delivery_days"] = (
        train_df[target_regression].values
    )

    train_features["late_delivery_flag"] = (
        train_df[target_classification].values
    )

    test_features["delivery_days"] = (
        test_df[target_regression].values
    )

    test_features["late_delivery_flag"] = (
        test_df[target_classification].values
    )

    # ------------------------------------------------------------------
    # Preserve order IDs separately for prediction traceability.
    # ------------------------------------------------------------------
    train_features.insert(
        0,
        "order_id",
        train_df["order_id"].values,
    )

    test_features.insert(
        0,
        "order_id",
        test_df["order_id"].values,
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    train_path = OUTPUT_DIR / "delivery_features_train.csv"
    test_path = OUTPUT_DIR / "delivery_features_test.csv"

    train_features.to_csv(
        train_path,
        index=False,
    )

    test_features.to_csv(
        test_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Validation summary.
    # ------------------------------------------------------------------
    print("\nSPLIT SUMMARY")
    print("-" * 70)

    print(f"Total orders:       {len(df):,}")
    print(f"Training orders:    {len(train_features):,}")
    print(f"Testing orders:     {len(test_features):,}")

    print(
        f"\nTraining period:    "
        f"{train_df['purchase_timestamp'].min()} "
        f"to "
        f"{train_df['purchase_timestamp'].max()}"
    )

    print(
        f"Testing period:     "
        f"{test_df['purchase_timestamp'].min()} "
        f"to "
        f"{test_df['purchase_timestamp'].max()}"
    )

    print(
        f"\nTraining late rate: "
        f"{train_features['late_delivery_flag'].mean() * 100:.2f}%"
    )

    print(
        f"Testing late rate:  "
        f"{test_features['late_delivery_flag'].mean() * 100:.2f}%"
    )

    print(
        f"\nFeature count:      "
        f"{len(train_features.columns) - 3}"
    )

    print(
        f"Train missing:      "
        f"{train_features.isna().sum().sum()}"
    )

    print(
        f"Test missing:       "
        f"{test_features.isna().sum().sum()}"
    )

    print(
        f"Train duplicate IDs:"
        f" {train_features['order_id'].duplicated().sum()}"
    )

    print(
        f"Test duplicate IDs: "
        f"{test_features['order_id'].duplicated().sum()}"
    )

    print("\nExcluded from model features:")
    print("  order_id")
    print("  purchase_timestamp")
    print("  delivered_date")
    print("  estimated_delivery_date")
    print("  delivery_days")
    print("  late_delivery_flag")
    print("  customer_key")

    print("\nSaved:")
    print(train_path)
    print(test_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
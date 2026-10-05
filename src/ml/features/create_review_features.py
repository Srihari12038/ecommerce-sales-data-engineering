from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "review_prediction_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
)


def main():
    print("=" * 70)
    print("CREATING REVIEW PREDICTION FEATURES")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["purchase_timestamp"],
    )

    # -----------------------------------------------------------------
    # Sort strictly by order purchase time.
    # -----------------------------------------------------------------
    df = (
        df
        .sort_values("purchase_timestamp")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Model target
    # -----------------------------------------------------------------
    target = "review_score"

    # -----------------------------------------------------------------
    # Numeric predictors
    #
    # customer_key is an identifier and is intentionally excluded.
    # -----------------------------------------------------------------
    numeric_features = [
        "item_count",
        "unique_products",
        "unique_sellers",
        "total_price",
        "total_freight",
        "total_order_value",
        "average_price",
        "average_freight",
        "average_product_name_length",
        "average_product_description_length",
        "average_product_photos",
        "average_product_weight_g",
        "max_product_weight_g",
        "average_product_volume_cm3",
        "same_state_ratio",
        "estimated_delivery_days",
        "payment_count",
        "total_payment_value",
        "max_installments",
        "average_installments",
        "purchase_year",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "freight_ratio",
        "payment_to_order_ratio",
    ]

    # -----------------------------------------------------------------
    # Categorical predictors
    # -----------------------------------------------------------------
    categorical_features = [
        "customer_state",
        "seller_state",
        "product_category",
        "payment_types",
    ]

    required_columns = (
        numeric_features
        + categorical_features
        + [
            "order_id",
            "purchase_timestamp",
            target,
        ]
    )

    df = df[required_columns].copy()

    # -----------------------------------------------------------------
    # Chronological 80/20 split
    # -----------------------------------------------------------------
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # -----------------------------------------------------------------
    # Train-only categorical encoding
    #
    # This prevents information from future/test categories from
    # entering the training representation.
    # -----------------------------------------------------------------
    train_encoded = pd.get_dummies(
        train_df[categorical_features],
        columns=categorical_features,
        dtype=int,
    )

    test_encoded = pd.get_dummies(
        test_df[categorical_features],
        columns=categorical_features,
        dtype=int,
    )

    # Test must use exactly the training feature space.
    test_encoded = test_encoded.reindex(
        columns=train_encoded.columns,
        fill_value=0,
    )

    # -----------------------------------------------------------------
    # Build feature matrices
    # -----------------------------------------------------------------
    train_features = pd.concat(
        [
            train_df[numeric_features].reset_index(drop=True),
            train_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    test_features = pd.concat(
        [
            test_df[numeric_features].reset_index(drop=True),
            test_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    # -----------------------------------------------------------------
    # Add target
    # -----------------------------------------------------------------
    train_features[target] = (
        train_df[target].values
    )

    test_features[target] = (
        test_df[target].values
    )

    # -----------------------------------------------------------------
    # Preserve order IDs for traceability
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    train_path = (
        OUTPUT_DIR
        / "review_features_train.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "review_features_test.csv"
    )

    train_features.to_csv(
        train_path,
        index=False,
    )

    test_features.to_csv(
        test_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    print("\nSPLIT SUMMARY")
    print("-" * 70)

    print(
        f"Total orders:       {len(df):,}"
    )

    print(
        f"Training orders:    {len(train_features):,}"
    )

    print(
        f"Testing orders:     {len(test_features):,}"
    )

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
        f"\nTraining average score: "
        f"{train_features[target].mean():.4f}"
    )

    print(
        f"Testing average score:  "
        f"{test_features[target].mean():.4f}"
    )

    print("\nTRAIN REVIEW DISTRIBUTION")
    print("-" * 70)

    print(
        (
            train_features[target]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print("\nTEST REVIEW DISTRIBUTION")
    print("-" * 70)

    print(
        (
            test_features[target]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print(
        f"\nFeature count:      "
        f"{len(train_features.columns) - 2}"
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

    print("\nEXCLUDED FROM MODEL FEATURES")
    print("-" * 70)
    print("order_id")
    print("customer_key")
    print("purchase_timestamp")
    print("review_score")

    print("\nSAVED")
    print("-" * 70)
    print(train_path)
    print(test_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
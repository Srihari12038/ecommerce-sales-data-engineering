from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# File location:
# repo/
# └── src/
#     └── ml/
#         └── features/
#             └── prepare_order_anomaly_features.py
#
# parents[3] = repository root
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "order_anomaly_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
)

OUTPUT_FEATURE_PATH = (
    OUTPUT_DIR
    / "order_anomaly_features.csv"
)

OUTPUT_STATISTICS_PATH = (
    OUTPUT_DIR
    / "order_anomaly_feature_statistics.csv"
)


def main():
    print("=" * 70)
    print("PREPARING ORDER ANOMALY FEATURES")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate input path before loading
    # -----------------------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    print("\nINPUT")
    print("-" * 70)
    print(INPUT_PATH)

    # -----------------------------------------------------------------
    # Load order-level anomaly dataset
    # -----------------------------------------------------------------
    df = pd.read_csv(
        INPUT_PATH
    )

    # -----------------------------------------------------------------
    # Features used for anomaly detection
    #
    # These represent transaction characteristics.
    # -----------------------------------------------------------------
    base_features = [
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

    # -----------------------------------------------------------------
    # Check required columns
    # -----------------------------------------------------------------
    missing_features = [
        feature
        for feature in base_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required anomaly features:\n"
            + "\n".join(missing_features)
        )

    # -----------------------------------------------------------------
    # Select anomaly features
    # -----------------------------------------------------------------
    features = df[
        base_features
    ].copy()

    # -----------------------------------------------------------------
    # Validate missing values
    # -----------------------------------------------------------------
    missing_count = int(
        features.isna().sum().sum()
    )

    if missing_count > 0:
        missing_by_column = (
            features
            .isna()
            .sum()
            .loc[lambda x: x > 0]
        )

        raise ValueError(
            "Missing values detected in anomaly features:\n"
            + missing_by_column.to_string()
        )

    # -----------------------------------------------------------------
    # Validate infinite values
    # -----------------------------------------------------------------
    numeric_array = features.to_numpy(
        dtype=float
    )

    infinite_count = int(
        np.isinf(numeric_array).sum()
    )

    if infinite_count > 0:
        raise ValueError(
            f"Detected {infinite_count} infinite values "
            "in anomaly features."
        )

    # -----------------------------------------------------------------
    # Validate negative values for variables that should be
    # non-negative.
    # -----------------------------------------------------------------
    non_negative_features = [
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
        "average_item_value",
    ]

    negative_summary = {}

    for feature in non_negative_features:
        negative_count = int(
            (features[feature] < 0).sum()
        )

        if negative_count > 0:
            negative_summary[feature] = negative_count

    if negative_summary:
        raise ValueError(
            "Negative values found in non-negative features:\n"
            + pd.Series(
                negative_summary
            ).to_string()
        )

    # -----------------------------------------------------------------
    # Log-transform heavily right-skewed transaction variables.
    #
    # log1p(x) = log(1 + x)
    #
    # This preserves zero and compresses large values.
    # -----------------------------------------------------------------
    log_features = [
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
        "average_item_value",
    ]

    transformed = pd.DataFrame(
        index=features.index
    )

    for feature in log_features:
        transformed[
            f"log1p_{feature}"
        ] = np.log1p(
            features[feature]
        )

    # -----------------------------------------------------------------
    # Keep ratio variables without log transformation.
    # -----------------------------------------------------------------
    transformed[
        "freight_ratio"
    ] = features[
        "freight_ratio"
    ]

    transformed[
        "payment_to_order_ratio"
    ] = features[
        "payment_to_order_ratio"
    ]

    # -----------------------------------------------------------------
    # Final validation after transformation
    # -----------------------------------------------------------------
    transformed_array = transformed.to_numpy(
        dtype=float
    )

    transformed_missing = int(
        transformed.isna().sum().sum()
    )

    transformed_infinite = int(
        np.isinf(transformed_array).sum()
    )

    if transformed_missing > 0:
        raise ValueError(
            "Missing values found after transformation."
        )

    if transformed_infinite > 0:
        raise ValueError(
            "Infinite values found after transformation."
        )

    # -----------------------------------------------------------------
    # RobustScaler
    #
    # RobustScaler uses median and IQR and is therefore appropriate
    # when extreme transaction values are present.
    # -----------------------------------------------------------------
    scaler = RobustScaler()

    scaled_values = scaler.fit_transform(
        transformed
    )

    scaled_columns = [
        f"scaled_{column}"
        for column in transformed.columns
    ]

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=scaled_columns,
        index=features.index,
    )

    # -----------------------------------------------------------------
    # Keep order identifiers only for traceability.
    #
    # They are NOT anomaly-model features.
    # -----------------------------------------------------------------
    traceability_columns = [
        "order_id",
        "order_purchase_timestamp",
        "order_status",
    ]

    missing_traceability = [
        column
        for column in traceability_columns
        if column not in df.columns
    ]

    if missing_traceability:
        raise ValueError(
            "Missing traceability columns:\n"
            + "\n".join(missing_traceability)
        )

    traceability = df[
        traceability_columns
    ].copy()

    # -----------------------------------------------------------------
    # Final feature dataset
    # -----------------------------------------------------------------
    result = pd.concat(
        [
            traceability.reset_index(drop=True),
            transformed.reset_index(drop=True),
            scaled_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # -----------------------------------------------------------------
    # Feature statistics
    # -----------------------------------------------------------------
    statistics = pd.DataFrame(
        {
            "feature": transformed.columns,
            "median": transformed.median().values,
            "q1": transformed.quantile(0.25).values,
            "q3": transformed.quantile(0.75).values,
            "mean": transformed.mean().values,
            "std": transformed.std().values,
        }
    )

    # -----------------------------------------------------------------
    # Final result validation
    # -----------------------------------------------------------------
    final_numeric = result.select_dtypes(
        include=np.number
    )

    final_missing = int(
        result.isna().sum().sum()
    )

    final_infinite = int(
        np.isinf(
            final_numeric.to_numpy()
        ).sum()
    )

    duplicate_orders = int(
        result["order_id"].duplicated().sum()
    )

    if final_missing > 0:
        raise ValueError(
            f"Final dataset contains {final_missing} missing values."
        )

    if final_infinite > 0:
        raise ValueError(
            f"Final dataset contains {final_infinite} infinite values."
        )

    if duplicate_orders > 0:
        raise ValueError(
            f"Final dataset contains {duplicate_orders} "
            "duplicate order IDs."
        )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FEATURE_PATH,
        index=False,
    )

    statistics.to_csv(
        OUTPUT_STATISTICS_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print("\nFEATURE SUMMARY")
    print("-" * 70)

    print(
        f"Orders:               {len(result):,}"
    )

    print(
        f"Original features:    {len(base_features)}"
    )

    print(
        f"Transformed features: {len(transformed.columns)}"
    )

    print(
        f"Scaled features:      {len(scaled_df.columns)}"
    )

    print(
        f"Total output columns: {len(result.columns)}"
    )

    print(
        f"Missing values:       {final_missing}"
    )

    print(
        f"Infinite values:      {final_infinite}"
    )

    print(
        f"Duplicate order IDs:  {duplicate_orders}"
    )

    print("\nTRANSFORMED FEATURES")
    print("-" * 70)

    for column in transformed.columns:
        print(column)

    print("\nTRACEABILITY COLUMNS")
    print("-" * 70)

    for column in traceability_columns:
        print(column)

    print("\nMODEL FEATURES EXCLUDED")
    print("-" * 70)
    print("order_id")
    print("order_purchase_timestamp")
    print("order_status")

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_FEATURE_PATH)
    print(OUTPUT_STATISTICS_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
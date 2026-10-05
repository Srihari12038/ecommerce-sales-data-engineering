import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_behavior_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_clustering_features.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    selected_features = [
        "recency",
        "monetary",
        "total_items",
        "total_freight",
        "unique_products",
        "unique_sellers",
        "average_review_score",
        "payment_count",
        "average_payment_value",
        "max_installments"
    ]

    clustering_data = df[
        ["customer_key"] + selected_features
    ].copy()

    clustering_data[selected_features] = (
        clustering_data[selected_features]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        clustering_data[selected_features]
    )

    scaled_columns = [
        f"{column}_scaled"
        for column in selected_features
    ]

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=scaled_columns
    )

    result = pd.concat(
        [
            clustering_data.reset_index(drop=True),
            scaled_df
        ],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 80)
    print("CUSTOMER CLUSTERING FEATURE DATASET")
    print("=" * 80)

    print(f"Customers: {len(result)}")
    print(
        f"Original features selected: "
        f"{len(selected_features)}"
    )
    print(
        f"Scaled features: "
        f"{len(scaled_columns)}"
    )

    print("\nSelected features")
    print("-" * 80)

    for feature in selected_features:
        print(feature)

    print("\nMissing values")
    print("-" * 80)

    print(
        result.isna()
        .sum()
        .to_string()
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
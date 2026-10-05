import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_clustering_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_segments.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "src"
    / "ml"
    / "models"
    / "customer_kmeans.pkl"
)


def main():

    # ================================================================
    # Load clustering dataset
    # ================================================================

    df = pd.read_csv(INPUT_FILE)

    feature_columns = [
        "recency_scaled",
        "monetary_scaled",
        "total_items_scaled",
        "total_freight_scaled",
        "unique_products_scaled",
        "unique_sellers_scaled",
        "average_review_score_scaled",
        "payment_count_scaled",
        "average_payment_value_scaled",
        "max_installments_scaled"
    ]

    X = df[feature_columns]

    # ================================================================
    # Train K-Means
    # ================================================================

    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X)

    # ================================================================
    # Create business-friendly segment labels
    # ================================================================

    cluster_summary = (
        df
        .groupby("cluster")[
            [
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
        ]
        .mean()
    )

    # Determine which cluster is more valuable
    high_value_cluster = (
        cluster_summary["monetary"]
        .idxmax()
    )

    df["segment"] = df["cluster"].apply(
        lambda x:
            "High Value Customers"
            if x == high_value_cluster
            else "Standard Customers"
    )

    # ================================================================
    # Save predictions
    # ================================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df[
        [
            "customer_key",
            "cluster",
            "segment",
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
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ================================================================
    # Save model
    # ================================================================

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ================================================================
    # Display cluster summary
    # ================================================================

    print("=" * 80)
    print("CUSTOMER SEGMENTATION MODEL")
    print("=" * 80)

    print("\nCluster Distribution")
    print("-" * 80)

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBusiness Segment Distribution")
    print("-" * 80)

    print(
        df["segment"]
        .value_counts()
        .to_string()
    )

    print("\nCluster Profiles")
    print("-" * 80)

    print(
        cluster_summary
        .round(2)
        .to_string()
    )

    print("\nHigh Value Cluster:")
    print(high_value_cluster)

    print("\nCustomer Segments:")
    print(OUTPUT_FILE)

    print("\nK-Means Model:")
    print(MODEL_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
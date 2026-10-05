import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    / "evaluation"
    / "cluster_evaluation.csv"
)

PLOT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "cluster_evaluation.png"
)


def main():

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

    results = []

    print("=" * 80)
    print("K-MEANS CLUSTER EVALUATION")
    print("=" * 80)

    for k in range(2, 9):

        print(f"\nTesting K = {k}")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X,
            labels
        )

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette
            }
        )

        print(
            f"Inertia: {inertia:.2f}"
        )

        print(
            f"Silhouette Score: {silhouette:.4f}"
        )

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ================================================================
    # Plot Elbow Curve
    # ================================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["k"],
        results_df["inertia"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Method")

    plt.xticks(
        results_df["k"]
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150
    )

    plt.close()

    # ================================================================
    # Results
    # ================================================================

    print("\n" + "=" * 80)
    print("CLUSTER EVALUATION RESULTS")
    print("=" * 80)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    best_k = (
        results_df
        .loc[
            results_df["silhouette_score"].idxmax(),
            "k"
        ]
    )

    best_score = (
        results_df
        ["silhouette_score"]
        .max()
    )

    print("\nBest K by Silhouette Score:")
    print(
        f"K = {int(best_k)}"
    )

    print(
        f"Silhouette Score = {best_score:.4f}"
    )

    print("\nEvaluation CSV:")
    print(OUTPUT_FILE)

    print("\nElbow Plot:")
    print(PLOT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
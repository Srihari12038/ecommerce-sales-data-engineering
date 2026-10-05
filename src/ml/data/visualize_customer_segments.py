import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_segments.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "visualizations"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ================================================================
    # 1. Customer count by segment
    # ================================================================

    segment_counts = (
        df["segment"]
        .value_counts()
    )

    plt.figure(figsize=(10, 6))

    segment_counts.plot(
        kind="bar"
    )

    plt.title(
        "Customer Count by Segment"
    )

    plt.xlabel(
        "Customer Segment"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "customer_count_by_segment.png",
        dpi=150
    )

    plt.close()

    # ================================================================
    # 2. Sales contribution
    # ================================================================

    sales = (
        df.groupby("segment")["monetary"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    sales.plot(
        kind="bar"
    )

    plt.title(
        "Sales Contribution by Customer Segment"
    )

    plt.xlabel(
        "Customer Segment"
    )

    plt.ylabel(
        "Total Sales"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sales_contribution_by_segment.png",
        dpi=150
    )

    plt.close()

    # ================================================================
    # 3. Monetary distribution
    # ================================================================

    plt.figure(figsize=(10, 6))

    df.boxplot(
        column="monetary",
        by="segment"
    )

    plt.title(
        "Monetary Value Distribution by Segment"
    )

    plt.suptitle("")

    plt.xlabel(
        "Customer Segment"
    )

    plt.ylabel(
        "Monetary Value"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monetary_distribution_by_segment.png",
        dpi=150
    )

    plt.close()

    # ================================================================
    # 4. Recency vs Monetary
    # ================================================================

    plt.figure(figsize=(10, 6))

    for segment in df["segment"].unique():

        subset = df[
            df["segment"] == segment
        ]

        plt.scatter(
            subset["recency"],
            subset["monetary"],
            label=segment,
            alpha=0.35
        )

    plt.title(
        "Customer Segmentation: Recency vs Monetary Value"
    )

    plt.xlabel(
        "Recency (Days)"
    )

    plt.ylabel(
        "Monetary Value"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "recency_vs_monetary.png",
        dpi=150
    )

    plt.close()

    print("=" * 80)
    print("CUSTOMER SEGMENT VISUALIZATIONS")
    print("=" * 80)

    print("\nGenerated visualizations:")

    print(
        OUTPUT_DIR
        / "customer_count_by_segment.png"
    )

    print(
        OUTPUT_DIR
        / "sales_contribution_by_segment.png"
    )

    print(
        OUTPUT_DIR
        / "monetary_distribution_by_segment.png"
    )

    print(
        OUTPUT_DIR
        / "recency_vs_monetary.png"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
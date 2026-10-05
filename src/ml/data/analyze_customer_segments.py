import pandas as pd

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_segments.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_segment_analysis.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    # ================================================================
    # Segment-level business metrics
    # ================================================================

    analysis = (
        df.groupby("segment")
        .agg(
            customers=("customer_key", "nunique"),
            average_recency_days=("recency", "mean"),
            median_recency_days=("recency", "median"),
            average_monetary_value=("monetary", "mean"),
            median_monetary_value=("monetary", "median"),
            total_sales=("monetary", "sum"),
            average_items=("total_items", "mean"),
            average_freight=("total_freight", "mean"),
            average_unique_products=("unique_products", "mean"),
            average_unique_sellers=("unique_sellers", "mean"),
            average_review_score=("average_review_score", "mean"),
            average_payment_value=("average_payment_value", "mean"),
            average_installments=("max_installments", "mean")
        )
        .reset_index()
    )

    # ================================================================
    # Customer percentage
    # ================================================================

    total_customers = analysis["customers"].sum()

    analysis["customer_percentage"] = (
        analysis["customers"]
        / total_customers
        * 100
    )

    # ================================================================
    # Revenue percentage
    # ================================================================

    total_sales = analysis["total_sales"].sum()

    analysis["sales_percentage"] = (
        analysis["total_sales"]
        / total_sales
        * 100
    )

    # ================================================================
    # Business recommendation
    # ================================================================

    analysis["business_action"] = analysis["segment"].map(
        {
            "High Value Customers":
                "Prioritize retention, premium offers, cross-selling and personalized campaigns.",

            "Standard Customers":
                "Use targeted promotions, product recommendations and incentives to increase order value."
        }
    )

    # ================================================================
    # Round numeric values
    # ================================================================

    numeric_columns = analysis.select_dtypes(
        include="number"
    ).columns

    analysis[numeric_columns] = (
        analysis[numeric_columns]
        .round(2)
    )

    # ================================================================
    # Save
    # ================================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    analysis.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ================================================================
    # Display report
    # ================================================================

    print("=" * 80)
    print("CUSTOMER SEGMENT BUSINESS ANALYSIS")
    print("=" * 80)

    print("\nSegment Analysis")
    print("-" * 80)

    print(
        analysis.to_string(
            index=False
        )
    )

    print("\nTotal Customers:")
    print(total_customers)

    print("\nTotal Sales:")
    print(round(total_sales, 2))

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
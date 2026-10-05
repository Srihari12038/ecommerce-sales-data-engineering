import sqlite3
import pandas as pd

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATABASE_FILE = (
    PROJECT_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

RFM_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_rfm.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "customer_behavior_features.csv"
)


def main():

    # ================================================================
    # 1. Load RFM dataset
    # ================================================================

    rfm = pd.read_csv(RFM_FILE)

    rfm["last_purchase_date"] = pd.to_datetime(
        rfm["last_purchase_date"]
    )

    # ================================================================
    # 2. Connect to warehouse
    # ================================================================

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # ================================================================
    # 3. Customer transaction behavior
    # ================================================================

    transaction_query = """
    SELECT
        f.customer_key,

        COUNT(*) AS total_items,

        SUM(f.price) AS total_sales,

        SUM(f.freight_value) AS total_freight,

        SUM(f.total_item_value) AS total_item_value,

        AVG(f.price) AS average_item_price,

        AVG(f.freight_value) AS average_freight,

        COUNT(DISTINCT f.product_key) AS unique_products,

        COUNT(DISTINCT f.seller_key) AS unique_sellers

    FROM fact_sales f

    GROUP BY f.customer_key
    """

    transactions = pd.read_sql_query(
        transaction_query,
        connection
    )

    # ================================================================
    # 4. Customer review behavior
    #
    # fact_reviews -> dim_order -> dim_customer
    # ================================================================

    review_query = """
    SELECT
        c.customer_key,

        AVG(r.review_score) AS average_review_score,

        COUNT(*) AS review_count

    FROM fact_reviews r

    JOIN dim_order o
        ON r.order_id = o.order_id

    JOIN dim_customer c
        ON o.customer_id = c.customer_id

    GROUP BY c.customer_key
    """

    reviews = pd.read_sql_query(
        review_query,
        connection
    )

    # ================================================================
    # 5. Customer payment behavior
    #
    # fact_payments -> dim_order -> dim_customer
    # ================================================================

    payment_query = """
    SELECT
        c.customer_key,

        COUNT(*) AS payment_count,

        AVG(p.payment_value) AS average_payment_value,

        MAX(p.payment_installments) AS max_installments

    FROM fact_payments p

    JOIN dim_order o
        ON p.order_id = o.order_id

    JOIN dim_customer c
        ON o.customer_id = c.customer_id

    GROUP BY c.customer_key
    """

    payments = pd.read_sql_query(
        payment_query,
        connection
    )

    connection.close()

    # ================================================================
    # 6. Merge all customer features
    # ================================================================

    features = rfm.merge(
        transactions,
        on="customer_key",
        how="left"
    )

    features = features.merge(
        reviews,
        on="customer_key",
        how="left"
    )

    features = features.merge(
        payments,
        on="customer_key",
        how="left"
    )

    # ================================================================
    # 7. Derived behavioral features
    # ================================================================

    features["average_order_value"] = (
        features["total_sales"]
        / features["frequency"]
    )

    features["items_per_order"] = (
        features["total_items"]
        / features["frequency"]
    )

    features["freight_ratio"] = (
        features["total_freight"]
        / features["total_sales"]
    )

    # ================================================================
    # 8. Handle missing numeric values
    # ================================================================

    numeric_columns = features.select_dtypes(
        include="number"
    ).columns

    features[numeric_columns] = (
        features[numeric_columns]
        .fillna(0)
    )

    # ================================================================
    # 9. Save dataset
    # ================================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ================================================================
    # 10. Display results
    # ================================================================

    print("=" * 80)
    print("CUSTOMER BEHAVIOR FEATURE DATASET")
    print("=" * 80)

    print(
        f"Customers: {len(features)}"
    )

    print(
        f"Features: {len(features.columns)}"
    )

    print("\nFeature columns")
    print("-" * 80)

    for column in features.columns:
        print(column)

    print("\nMissing values")
    print("-" * 80)

    print(
        features.isna()
        .sum()
        .to_string()
    )

    print("\nDuplicate customers")
    print("-" * 80)

    print(
        features["customer_key"]
        .duplicated()
        .sum()
    )

    print("\nFeature statistics")
    print("-" * 80)

    print(
        features[
            [
                "recency",
                "frequency",
                "monetary",
                "total_items",
                "total_sales",
                "total_freight",
                "unique_products",
                "unique_sellers",
                "average_review_score",
                "average_order_value"
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
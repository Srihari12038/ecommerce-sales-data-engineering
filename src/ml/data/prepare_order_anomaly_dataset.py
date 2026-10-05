from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

DB_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "order_anomaly_dataset.csv"
)


def main():
    print("=" * 70)
    print("PREPARING ORDER-LEVEL ANOMALY DATASET")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # -----------------------------------------------------------------
    # Aggregate sales transactions to one row per order.
    # -----------------------------------------------------------------
    sales_query = """
    SELECT
        f.order_id,

        COUNT(*) AS item_count,
        COUNT(DISTINCT f.product_key) AS unique_products,
        COUNT(DISTINCT f.seller_key) AS unique_sellers,

        SUM(f.price) AS total_price,
        SUM(f.freight_value) AS total_freight,
        SUM(f.total_item_value) AS total_order_value,

        AVG(f.price) AS average_price,
        AVG(f.freight_value) AS average_freight

    FROM fact_sales f

    GROUP BY f.order_id
    """

    orders = pd.read_sql_query(
        sales_query,
        conn,
    )

    # -----------------------------------------------------------------
    # Payment aggregation.
    # -----------------------------------------------------------------
    payment_query = """
    SELECT
        order_id,

        COUNT(*) AS payment_count,

        SUM(payment_value) AS total_payment_value,

        MAX(payment_installments) AS max_installments,

        AVG(payment_installments) AS average_installments

    FROM fact_payments

    GROUP BY order_id
    """

    payments = pd.read_sql_query(
        payment_query,
        conn,
    )

    # -----------------------------------------------------------------
    # Order timing and status.
    # -----------------------------------------------------------------
    date_query = """
    SELECT
        order_id,
        order_purchase_timestamp,
        order_status
    FROM dim_order
    WHERE order_purchase_timestamp IS NOT NULL
    """

    order_dates = pd.read_sql_query(
        date_query,
        conn,
    )

    conn.close()

    # -----------------------------------------------------------------
    # Merge sales + payment + order information.
    # -----------------------------------------------------------------
    orders = orders.merge(
        payments,
        on="order_id",
        how="left",
    )

    orders = orders.merge(
        order_dates,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # Parse purchase timestamp.
    # -----------------------------------------------------------------
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    # -----------------------------------------------------------------
    # Time features.
    # -----------------------------------------------------------------
    orders["purchase_year"] = (
        orders["order_purchase_timestamp"].dt.year
    )

    orders["purchase_month"] = (
        orders["order_purchase_timestamp"].dt.month
    )

    orders["purchase_dayofweek"] = (
        orders["order_purchase_timestamp"].dt.dayofweek
    )

    orders["purchase_hour"] = (
        orders["order_purchase_timestamp"].dt.hour
    )

    # -----------------------------------------------------------------
    # Fill missing payment information.
    #
    # Some orders have no corresponding payment record. For anomaly
    # feature construction, those payment aggregates are treated as 0.
    # -----------------------------------------------------------------
    payment_columns = [
        "payment_count",
        "total_payment_value",
        "max_installments",
        "average_installments",
    ]

    for column in payment_columns:
        orders[column] = (
            orders[column]
            .fillna(0)
        )

    # -----------------------------------------------------------------
    # Derived financial features.
    # -----------------------------------------------------------------
    orders["freight_ratio"] = np.where(
        orders["total_price"] > 0,
        orders["total_freight"]
        / orders["total_price"],
        0,
    )

    orders["payment_to_order_ratio"] = np.where(
        orders["total_order_value"] > 0,
        orders["total_payment_value"]
        / orders["total_order_value"],
        0,
    )

    orders["average_item_value"] = np.where(
        orders["item_count"] > 0,
        orders["total_order_value"]
        / orders["item_count"],
        0,
    )

    # -----------------------------------------------------------------
    # Explicitly handle any remaining missing payment ratio.
    #
    # Example:
    # total_order_value > 0 but total_payment_value = 0
    # => payment_to_order_ratio = 0
    #
    # This retains the order instead of dropping it.
    # -----------------------------------------------------------------
    orders["payment_to_order_ratio"] = (
        orders["payment_to_order_ratio"]
        .fillna(0)
    )

    # -----------------------------------------------------------------
    # Validate required fields before saving.
    # -----------------------------------------------------------------
    required_columns = [
        "order_id",
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
        "order_purchase_timestamp",
        "order_status",
    ]

    missing_required_columns = [
        column
        for column in required_columns
        if column not in orders.columns
    ]

    if missing_required_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required_columns)
        )

    # -----------------------------------------------------------------
    # Sort chronologically.
    # -----------------------------------------------------------------
    orders = (
        orders
        .sort_values(
            "order_purchase_timestamp"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Final missing-value check.
    # -----------------------------------------------------------------
    missing_total = int(
        orders.isna().sum().sum()
    )

    if missing_total > 0:
        missing_by_column = (
            orders.isna().sum()
            .loc[lambda x: x > 0]
        )

        raise ValueError(
            "Dataset still contains missing values:\n"
            + missing_by_column.to_string()
        )

    # -----------------------------------------------------------------
    # Validation summary.
    # -----------------------------------------------------------------
    print("\nDATASET SUMMARY")
    print("-" * 70)

    print(
        f"Orders:              "
        f"{len(orders):,}"
    )

    print(
        f"Columns:             "
        f"{len(orders.columns)}"
    )

    print(
        f"Duplicate order IDs: "
        f"{orders['order_id'].duplicated().sum()}"
    )

    print(
        f"Missing values:      "
        f"{missing_total}"
    )

    print(
        f"Purchase date range: "
        f"{orders['order_purchase_timestamp'].min()} "
        f"to "
        f"{orders['order_purchase_timestamp'].max()}"
    )

    print("\nFINANCIAL SUMMARY")
    print("-" * 70)

    financial_columns = [
        "total_price",
        "total_freight",
        "total_order_value",
        "item_count",
        "unique_products",
        "unique_sellers",
        "payment_count",
        "total_payment_value",
        "freight_ratio",
        "payment_to_order_ratio",
    ]

    for column in financial_columns:
        print(
            f"{column:<25}"
            f" mean={orders[column].mean():.4f}"
            f" median={orders[column].median():.4f}"
            f" max={orders[column].max():.4f}"
        )

    print("\nFEATURES FOR ANOMALY DETECTION")
    print("-" * 70)

    anomaly_features = [
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

    for feature in anomaly_features:
        print(feature)

    # -----------------------------------------------------------------
    # Save dataset.
    # -----------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    orders.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
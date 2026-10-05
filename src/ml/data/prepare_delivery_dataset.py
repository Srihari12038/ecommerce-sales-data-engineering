from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DB_PATH = BASE_DIR / "data" / "database" / "ecommerce.db"
OUTPUT_PATH = BASE_DIR / "data" / "ml" / "delivery_prediction_dataset.csv"


def main():
    print("=" * 70)
    print("PREPARING DELIVERY PREDICTION DATASET")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        f.order_id,
        f.order_item_id,
        f.customer_key,
        f.product_key,
        f.seller_key,

        f.price,
        f.freight_value,
        f.total_item_value,

        c.customer_state,

        s.seller_state,

        p.product_category_name_english,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm,
        p.product_photos_qty,

        o.order_purchase_timestamp,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date

    FROM fact_sales f

    INNER JOIN dim_customer c
        ON f.customer_key = c.customer_key

    INNER JOIN dim_seller s
        ON f.seller_key = s.seller_key

    LEFT JOIN dim_product p
        ON f.product_key = p.product_key

    INNER JOIN dim_order o
        ON f.order_id = o.order_id

    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_purchase_timestamp IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)

    # Payment information is available before delivery,
    # so it can safely be used as a predictive feature.
    payment_query = """
    SELECT
        order_id,
        COUNT(*) AS payment_count,
        SUM(payment_value) AS total_payment_value,
        MAX(payment_installments) AS max_installments
    FROM fact_payments
    GROUP BY order_id
    """

    payments = pd.read_sql_query(payment_query, conn)

    conn.close()

    print(f"Item-level rows loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # Date/time preparation
    # ------------------------------------------------------------------
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"],
        errors="coerce"
    )

    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce"
    )

    # Actual target.
    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Estimated delivery duration known at order time.
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # ------------------------------------------------------------------
    # Item-level derived features
    # ------------------------------------------------------------------
    df["product_volume_cm3"] = (
        df["product_length_cm"]
        * df["product_height_cm"]
        * df["product_width_cm"]
    )

    df["same_state"] = (
        df["customer_state"] == df["seller_state"]
    ).astype(int)

    # ------------------------------------------------------------------
    # Aggregate item-level records into one row per order
    # ------------------------------------------------------------------
    order_df = (
        df.groupby("order_id")
        .agg(
            customer_key=("customer_key", "first"),

            item_count=("order_item_id", "count"),
            unique_products=("product_key", "nunique"),
            unique_sellers=("seller_key", "nunique"),

            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            total_order_value=("total_item_value", "sum"),

            average_price=("price", "mean"),
            average_freight=("freight_value", "mean"),

            average_product_weight_g=("product_weight_g", "mean"),
            max_product_weight_g=("product_weight_g", "max"),

            average_product_volume_cm3=("product_volume_cm3", "mean"),
            average_product_photos=("product_photos_qty", "mean"),

            same_state_ratio=("same_state", "mean"),

            customer_state=("customer_state", "first"),

            purchase_timestamp=("order_purchase_timestamp", "first"),
            delivery_days=("delivery_days", "first"),
            estimated_delivery_days=("estimated_delivery_days", "first"),
            delivered_date=("order_delivered_customer_date", "first"),
            estimated_delivery_date=("order_estimated_delivery_date", "first"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Payment features
    # ------------------------------------------------------------------
    order_df = order_df.merge(
        payments,
        on="order_id",
        how="left"
    )

    # ------------------------------------------------------------------
    # Time features known at purchase time
    # ------------------------------------------------------------------
    order_df["purchase_year"] = order_df["purchase_timestamp"].dt.year
    order_df["purchase_month"] = order_df["purchase_timestamp"].dt.month
    order_df["purchase_dayofweek"] = (
        order_df["purchase_timestamp"].dt.dayofweek
    )
    order_df["purchase_hour"] = order_df["purchase_timestamp"].dt.hour

    # ------------------------------------------------------------------
    # Business features
    # ------------------------------------------------------------------
    order_df["freight_ratio"] = np.where(
        order_df["total_price"] > 0,
        order_df["total_freight"] / order_df["total_price"],
        0
    )

    # Late-delivery target.
    order_df["late_delivery_flag"] = (
        order_df["delivery_days"]
        > order_df["estimated_delivery_days"]
    ).astype(int)

    # ------------------------------------------------------------------
    # Remove invalid target rows
    # ------------------------------------------------------------------
    order_df = order_df[
        order_df["delivery_days"].notna()
        & (order_df["delivery_days"] >= 0)
    ].copy()

    # ------------------------------------------------------------------
    # Fill missing predictive values
    # ------------------------------------------------------------------
    numeric_columns = [
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
    ]

    for column in numeric_columns:
        if column in order_df.columns:
            order_df[column] = order_df[column].fillna(
                order_df[column].median()
            )

    order_df["customer_state"] = (
        order_df["customer_state"]
        .fillna("Unknown")
    )

    # ------------------------------------------------------------------
    # Sort by purchase time
    # ------------------------------------------------------------------
    order_df = order_df.sort_values(
        "purchase_timestamp"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    order_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("\nDATASET SUMMARY")
    print("-" * 70)
    print(f"Rows:               {len(order_df):,}")
    print(f"Columns:            {len(order_df.columns)}")
    print(
        f"Date range:         "
        f"{order_df['purchase_timestamp'].min().date()} "
        f"to "
        f"{order_df['purchase_timestamp'].max().date()}"
    )
    print(
        f"Mean delivery days: "
        f"{order_df['delivery_days'].mean():.2f}"
    )
    print(
        f"Median delivery days:"
        f" {order_df['delivery_days'].median():.2f}"
    )
    print(
        f"Late delivery rate: "
        f"{order_df['late_delivery_flag'].mean() * 100:.2f}%"
    )
    print(
        f"Missing values:     "
        f"{order_df.isna().sum().sum()}"
    )
    print(
        f"Duplicate orders:   "
        f"{order_df['order_id'].duplicated().sum()}"
    )

    print("\nCOLUMNS")
    print("-" * 70)
    for column in order_df.columns:
        print(column)

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
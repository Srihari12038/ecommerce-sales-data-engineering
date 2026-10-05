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
    / "review_prediction_dataset.csv"
)


def main():
    print("=" * 70)
    print("PREPARING REVIEW SCORE PREDICTION DATASET")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # -----------------------------------------------------------------
    # Review targets
    #
    # We deliberately use only review_score.
    # Review text/timestamps are NOT loaded because they occur after
    # the customer experience.
    # -----------------------------------------------------------------
    review_query = """
    SELECT
        order_id,
        review_id,
        review_score
    FROM fact_reviews
    WHERE review_score IS NOT NULL
    """

    reviews = pd.read_sql_query(
        review_query,
        conn,
    )

    print(f"Review rows loaded: {len(reviews):,}")

    # -----------------------------------------------------------------
    # Inspect multiple reviews per order.
    #
    # Ideally there is one review per order. If multiple rows exist,
    # choose the mode. If there is a tie, choose the first mode value.
    # -----------------------------------------------------------------
    review_counts = (
        reviews
        .groupby("order_id")
        .size()
        .rename("review_count")
        .reset_index()
    )

    duplicate_review_orders = (
        review_counts[
            review_counts["review_count"] > 1
        ]
    )

    print(
        f"Orders with multiple review rows: "
        f"{len(duplicate_review_orders):,}"
    )

    reviews = (
        reviews
        .groupby("order_id")
        .agg(
            review_score=(
                "review_score",
                lambda x: x.mode().iloc[0]
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Order + sales + customer + product + seller information
    #
    # Only information available before the review is created should
    # become model features.
    # -----------------------------------------------------------------
    order_query = """
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
        p.product_name_length,
        p.product_description_length,
        p.product_photos_qty,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm,

        o.order_purchase_timestamp,
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

    WHERE o.order_purchase_timestamp IS NOT NULL
    """

    sales = pd.read_sql_query(
        order_query,
        conn,
    )

    print(
        f"Order-item rows loaded: "
        f"{len(sales):,}"
    )

    # -----------------------------------------------------------------
    # Payment information
    #
    # Payments are available before the review and can therefore be
    # used as predictive features.
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

    payment_type_query = """
    SELECT
        order_id,
        payment_type
    FROM fact_payments
    GROUP BY order_id, payment_type
    """

    payment_types = pd.read_sql_query(
        payment_type_query,
        conn,
    )

    conn.close()

    # -----------------------------------------------------------------
    # Payment type consolidation
    # -----------------------------------------------------------------
    payment_type_summary = (
        payment_types
        .groupby("order_id")["payment_type"]
        .apply(
            lambda x: "|".join(
                sorted(
                    set(
                        x.dropna()
                        .astype(str)
                    )
                )
            )
        )
        .rename("payment_types")
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Parse dates
    # -----------------------------------------------------------------
    sales["order_purchase_timestamp"] = pd.to_datetime(
        sales["order_purchase_timestamp"],
        errors="coerce",
    )

    sales["order_estimated_delivery_date"] = pd.to_datetime(
        sales["order_estimated_delivery_date"],
        errors="coerce",
    )

    # -----------------------------------------------------------------
    # Estimated delivery duration
    #
    # This is known at order time and therefore is not leakage.
    # -----------------------------------------------------------------
    sales["estimated_delivery_days"] = (
        sales["order_estimated_delivery_date"]
        - sales["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # -----------------------------------------------------------------
    # Product volume
    # -----------------------------------------------------------------
    sales["product_volume_cm3"] = (
        sales["product_length_cm"]
        * sales["product_height_cm"]
        * sales["product_width_cm"]
    )

    # -----------------------------------------------------------------
    # Customer/seller geographic relationship
    # -----------------------------------------------------------------
    sales["same_state"] = (
        sales["customer_state"]
        == sales["seller_state"]
    ).astype(int)

    # -----------------------------------------------------------------
    # Aggregate item-level records to order level
    # -----------------------------------------------------------------
    orders = (
        sales
        .groupby("order_id")
        .agg(
            customer_key=("customer_key", "first"),

            item_count=("order_item_id", "count"),
            unique_products=("product_key", "nunique"),
            unique_sellers=("seller_key", "nunique"),

            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            total_order_value=(
                "total_item_value",
                "sum",
            ),

            average_price=("price", "mean"),
            average_freight=("freight_value", "mean"),

            average_product_name_length=(
                "product_name_length",
                "mean",
            ),

            average_product_description_length=(
                "product_description_length",
                "mean",
            ),

            average_product_photos=(
                "product_photos_qty",
                "mean",
            ),

            average_product_weight_g=(
                "product_weight_g",
                "mean",
            ),

            max_product_weight_g=(
                "product_weight_g",
                "max",
            ),

            average_product_volume_cm3=(
                "product_volume_cm3",
                "mean",
            ),

            same_state_ratio=(
                "same_state",
                "mean",
            ),

            customer_state=(
                "customer_state",
                "first",
            ),

            seller_state=(
                "seller_state",
                "first",
            ),

            product_category=(
                "product_category_name_english",
                "first",
            ),

            purchase_timestamp=(
                "order_purchase_timestamp",
                "first",
            ),

            estimated_delivery_days=(
                "estimated_delivery_days",
                "first",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Payment features
    # -----------------------------------------------------------------
    orders = orders.merge(
        payments,
        on="order_id",
        how="left",
    )

    orders = orders.merge(
        payment_type_summary,
        on="order_id",
        how="left",
    )

    # -----------------------------------------------------------------
    # Review target
    # -----------------------------------------------------------------
    orders = orders.merge(
        reviews,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # Calendar features known at purchase time
    # -----------------------------------------------------------------
    orders["purchase_year"] = (
        orders["purchase_timestamp"].dt.year
    )

    orders["purchase_month"] = (
        orders["purchase_timestamp"].dt.month
    )

    orders["purchase_dayofweek"] = (
        orders["purchase_timestamp"].dt.dayofweek
    )

    orders["purchase_hour"] = (
        orders["purchase_timestamp"].dt.hour
    )

    # -----------------------------------------------------------------
    # Derived business features
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

    # -----------------------------------------------------------------
    # Fill missing categorical values
    # -----------------------------------------------------------------
    categorical_columns = [
        "customer_state",
        "seller_state",
        "product_category",
        "payment_types",
    ]

    for column in categorical_columns:
        orders[column] = (
            orders[column]
            .fillna("Unknown")
            .astype(str)
        )

    # -----------------------------------------------------------------
    # Fill missing numerical values using medians.
    # -----------------------------------------------------------------
    numeric_columns = [
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
        "freight_ratio",
        "payment_to_order_ratio",
    ]

    for column in numeric_columns:
        if column in orders.columns:
            median_value = orders[column].median()

            orders[column] = orders[column].fillna(
                median_value
            )

    # -----------------------------------------------------------------
    # Remove invalid target rows.
    # -----------------------------------------------------------------
    orders = orders[
        orders["review_score"].between(
            1,
            5,
        )
    ].copy()

    # -----------------------------------------------------------------
    # Sort chronologically.
    # -----------------------------------------------------------------
    orders = orders.sort_values(
        "purchase_timestamp"
    ).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    print("\nDATASET SUMMARY")
    print("-" * 70)

    print(
        f"Orders with reviews: "
        f"{len(orders):,}"
    )

    print(
        f"Columns: "
        f"{len(orders.columns)}"
    )

    print(
        f"Date range: "
        f"{orders['purchase_timestamp'].min()} "
        f"to "
        f"{orders['purchase_timestamp'].max()}"
    )

    print(
        f"Missing values: "
        f"{orders.isna().sum().sum()}"
    )

    print(
        f"Duplicate order IDs: "
        f"{orders['order_id'].duplicated().sum()}"
    )

    print("\nREVIEW SCORE DISTRIBUTION")
    print("-" * 70)

    print(
        orders["review_score"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nREVIEW SCORE PERCENTAGE")
    print("-" * 70)

    print(
        (
            orders["review_score"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )

    print("\nAVERAGE REVIEW SCORE")
    print("-" * 70)

    print(
        f"{orders['review_score'].mean():.4f}"
    )

    # -----------------------------------------------------------------
    # Explicitly show fields that are NOT included in the final
    # predictive dataset.
    # -----------------------------------------------------------------
    excluded_from_prediction = [
        "review_id",
        "review_creation_date",
        "review_answer_timestamp",
        "review_comment_title",
        "review_comment_message",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "delivery_days",
    ]

    print("\nPOST-EXPERIENCE FIELDS EXCLUDED")
    print("-" * 70)

    for column in excluded_from_prediction:
        print(column)

    # -----------------------------------------------------------------
    # Save
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
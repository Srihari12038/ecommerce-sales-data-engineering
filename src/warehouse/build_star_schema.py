from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

CLEAN_DIR = Path("data/warehouse")
RAW_DIR = Path("data/raw")
WAREHOUSE_DIR = Path("data/warehouse")

# ============================================================
# LOAD CSV DATA
# ============================================================

def load_csv(filename):
    return pd.read_csv(CLEAN_DIR / filename)


# ============================================================
# DIMENSION: CUSTOMER
# ============================================================

def build_dim_customer(customers):
    df = customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ]
    ].copy()

    df.insert(
        0,
        "customer_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# DIMENSION: PRODUCT
# ============================================================

def build_dim_product(products, translation):
    df = products.copy()

    # Add English product category
    df = df.merge(
        translation,
        on="product_category_name",
        how="left",
    )

    df = df[
        [
            "product_id",
            "product_category_name",
            "product_category_name_english",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ].copy()

    # Correct spelling from original Olist dataset
    df = df.rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    )

    df.insert(
        0,
        "product_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# DIMENSION: SELLER
# ============================================================

def build_dim_seller(sellers):
    df = sellers[
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ]
    ].copy()

    df.insert(
        0,
        "seller_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# DIMENSION: DATE
# ============================================================

def build_dim_date(orders):
    dates = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    ).dt.normalize()

    min_date = dates.min()
    max_date = dates.max()

    date_range = pd.date_range(
        start=min_date,
        end=max_date,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "full_date": date_range
        }
    )

    df["date_key"] = (
        df["full_date"]
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    df["year"] = df["full_date"].dt.year
    df["quarter"] = df["full_date"].dt.quarter
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()

    df["week"] = (
        df["full_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day"] = df["full_date"].dt.day
    df["day_name"] = df["full_date"].dt.day_name()

    return df[
        [
            "date_key",
            "full_date",
            "year",
            "quarter",
            "month",
            "month_name",
            "week",
            "day",
            "day_name",
        ]
    ]


# ============================================================
# DIMENSION: ORDER
# ============================================================

def build_dim_order(orders):
    df = orders[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "delivery_days",
        ]
    ].copy()

    df.insert(
        0,
        "order_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# FACT: SALES
# ============================================================

def build_fact_sales(
    order_items,
    orders,
    customer_dim,
    product_dim,
    seller_dim,
):
    # Connect order items with orders
    df = order_items.merge(
        orders[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
            ]
        ],
        on="order_id",
        how="inner",
    )

    # Add customer surrogate key
    df = df.merge(
        customer_dim[
            [
                "customer_key",
                "customer_id",
            ]
        ],
        on="customer_id",
        how="left",
    )

    # Add product surrogate key
    df = df.merge(
        product_dim[
            [
                "product_key",
                "product_id",
            ]
        ],
        on="product_id",
        how="left",
    )

    # Add seller surrogate key
    df = df.merge(
        seller_dim[
            [
                "seller_key",
                "seller_id",
            ]
        ],
        on="seller_id",
        how="left",
    )

    # Convert timestamp
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    # Create date key
    df["order_date_key"] = (
        df["order_purchase_timestamp"]
        .dt.strftime("%Y%m%d")
        .astype("Int64")
    )

    # Calculate total item value
    df["total_item_value"] = (
        df["price"].fillna(0)
        + df["freight_value"].fillna(0)
    )

    # Create sales surrogate key
    df.insert(
        0,
        "sales_key",
        range(1, len(df) + 1),
    )

    return df[
        [
            "sales_key",
            "order_id",
            "order_item_id",
            "customer_key",
            "product_key",
            "seller_key",
            "order_date_key",
            "price",
            "freight_value",
            "total_item_value",
        ]
    ]


# ============================================================
# FACT: PAYMENTS
# ============================================================

def build_fact_payments(payments):
    df = payments[
        [
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
            "payment_value",
        ]
    ].copy()

    df.insert(
        0,
        "payment_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# FACT: REVIEWS
# ============================================================

def build_fact_reviews(reviews):
    df = reviews[
        [
            "review_id",
            "order_id",
            "review_score",
            "review_creation_date",
            "review_answer_timestamp",
        ]
    ].copy()

    df.insert(
        0,
        "review_key",
        range(1, len(df) + 1),
    )

    return df


# ============================================================
# SAVE TABLE
# ============================================================

def save_table(df, filename):
    output_path = WAREHOUSE_DIR / filename

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {filename}: {len(df):,} rows"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Create warehouse directory
    WAREHOUSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("BUILDING OLIST STAR SCHEMA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    print("\nLoading processed data...")

    customers = load_csv(
        "customers_clean.csv"
    )

    products = load_csv(
        "products_clean.csv"
    )

    sellers = load_csv(
        "sellers_clean.csv"
    )

    orders = load_csv(
        "orders_clean.csv"
    )

    order_items = load_csv(
        "order_items_clean.csv"
    )

    payments = load_csv(
        "payments_clean.csv"
    )

    # --------------------------------------------------------
    # Load raw data not transformed yet
    # --------------------------------------------------------

    reviews = pd.read_csv(
        RAW_DIR / "olist_order_reviews_dataset.csv"
    )

    translation = pd.read_csv(
        RAW_DIR / "product_category_name_translation.csv"
    )

    # --------------------------------------------------------
    # Build dimensions
    # --------------------------------------------------------

    print("\nBuilding dimensions...")

    dim_customer = build_dim_customer(
        customers
    )

    dim_product = build_dim_product(
        products,
        translation,
    )

    dim_seller = build_dim_seller(
        sellers
    )

    dim_date = build_dim_date(
        orders
    )

    dim_order = build_dim_order(
        orders
    )

    # --------------------------------------------------------
    # Build fact tables
    # --------------------------------------------------------

    print("\nBuilding fact tables...")

    fact_sales = build_fact_sales(
        order_items,
        orders,
        dim_customer,
        dim_product,
        dim_seller,
    )

    fact_payments = build_fact_payments(
        payments
    )

    fact_reviews = build_fact_reviews(
        reviews
    )

    # --------------------------------------------------------
    # Save dimension tables
    # --------------------------------------------------------

    print("\nSaving warehouse tables...")

    save_table(
        dim_customer,
        "dim_customer.csv",
    )

    save_table(
        dim_product,
        "dim_product.csv",
    )

    save_table(
        dim_seller,
        "dim_seller.csv",
    )

    save_table(
        dim_date,
        "dim_date.csv",
    )

    save_table(
        dim_order,
        "dim_order.csv",
    )

    # --------------------------------------------------------
    # Save fact tables
    # --------------------------------------------------------

    save_table(
        fact_sales,
        "fact_sales.csv",
    )

    save_table(
        fact_payments,
        "fact_payments.csv",
    )

    save_table(
        fact_reviews,
        "fact_reviews.csv",
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAR SCHEMA GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
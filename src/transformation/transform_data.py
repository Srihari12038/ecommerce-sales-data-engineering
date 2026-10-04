from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
WAREHOUSE_DIR = Path("data/warehouse")


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(filename):
    return pd.read_csv(PROCESSED_DIR / filename)


# ============================================================
# COMMON CLEANING
# ============================================================

def clean_column_names(df):
    """
    Standardize column names:
    - Remove leading/trailing spaces
    - Convert to lowercase
    - Replace spaces with underscores
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# ORDERS
# ============================================================

def transform_orders(df):
    df = clean_column_names(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Convert date/time columns
    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    # Create date attributes
    df["order_year"] = (
        df["order_purchase_timestamp"]
        .dt.year
    )

    df["order_month"] = (
        df["order_purchase_timestamp"]
        .dt.month
    )

    df["order_quarter"] = (
        df["order_purchase_timestamp"]
        .dt.quarter
    )

    df["order_date"] = (
        df["order_purchase_timestamp"]
        .dt.date
    )

    # Calculate delivery duration
    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    output_path = WAREHOUSE_DIR / "orders_clean.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# ORDER ITEMS
# ============================================================

def transform_order_items(df):
    df = clean_column_names(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Convert numeric columns
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    )

    df["freight_value"] = pd.to_numeric(
        df["freight_value"],
        errors="coerce",
    )

    output_path = (
        WAREHOUSE_DIR / "order_items_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# CUSTOMERS
# ============================================================

def transform_customers(df):
    df = clean_column_names(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    output_path = (
        WAREHOUSE_DIR / "customers_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# PRODUCTS
# ============================================================

def transform_products(df):
    df = clean_column_names(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    output_path = (
        WAREHOUSE_DIR / "products_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# SELLERS
# ============================================================

def transform_sellers(df):
    df = clean_column_names(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    output_path = (
        WAREHOUSE_DIR / "sellers_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# PAYMENTS
# ============================================================

def transform_payments(df):
    df = clean_column_names(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Convert numeric columns
    df["payment_installments"] = pd.to_numeric(
        df["payment_installments"],
        errors="coerce",
    )

    df["payment_value"] = pd.to_numeric(
        df["payment_value"],
        errors="coerce",
    )

    output_path = (
        WAREHOUSE_DIR / "payments_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created {output_path}: {len(df):,} rows"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directory
    WAREHOUSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("STARTING DATA TRANSFORMATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load source datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    orders = load_csv(
        "olist_orders_dataset.csv"
    )

    order_items = load_csv(
        "olist_order_items_dataset.csv"
    )

    customers = load_csv(
        "olist_customers_dataset.csv"
    )

    products = load_csv(
        "olist_products_dataset.csv"
    )

    sellers = load_csv(
        "olist_sellers_dataset.csv"
    )

    payments = load_csv(
        "olist_order_payments_dataset.csv"
    )

    # --------------------------------------------------------
    # Transform datasets
    # --------------------------------------------------------

    print("\nTransforming orders...")

    transform_orders(
        orders
    )

    print("\nTransforming order items...")

    transform_order_items(
        order_items
    )

    print("\nTransforming customers...")

    transform_customers(
        customers
    )

    print("\nTransforming products...")

    transform_products(
        products
    )

    print("\nTransforming sellers...")

    transform_sellers(
        sellers
    )

    print("\nTransforming payments...")

    transform_payments(
        payments
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
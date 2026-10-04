import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
WAREHOUSE_DIR = Path("data/warehouse")


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def transform_orders():
    input_file = PROCESSED_DIR / "olist_orders_dataset.csv"
    output_file = WAREHOUSE_DIR / "orders_clean.csv"

    df = pd.read_csv(input_file)

    original_rows = len(df)

    # Standardize column names
    df = clean_column_names(df)

    # Convert date columns
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Derived date fields
    df["order_year"] = df["order_purchase_timestamp"].dt.year
    df["order_month"] = df["order_purchase_timestamp"].dt.month
    df["order_quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["order_date"] = df["order_purchase_timestamp"].dt.date

    # Delivery duration
    if (
        "order_delivered_customer_date" in df.columns
        and "order_purchase_timestamp" in df.columns
    ):
        df["delivery_days"] = (
            df["order_delivered_customer_date"]
            - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Orders transformation complete")
    print(f"Original rows : {original_rows:,}")
    print(f"Final rows    : {len(df):,}")
    print(f"Output        : {output_file}")


def transform_order_items():
    input_file = PROCESSED_DIR / "olist_order_items_dataset.csv"
    output_file = WAREHOUSE_DIR / "order_items_clean.csv"

    df = pd.read_csv(input_file)

    df = clean_column_names(df)
    df = df.drop_duplicates()

    # Convert numeric columns
    for column in ["price", "freight_value"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Order items transformation complete")
    print(f"Rows          : {len(df):,}")
    print(f"Output        : {output_file}")


def transform_customers():
    input_file = PROCESSED_DIR / "olist_customers_dataset.csv"
    output_file = WAREHOUSE_DIR / "customers_clean.csv"

    df = pd.read_csv(input_file)

    df = clean_column_names(df)
    df = df.drop_duplicates()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Customers transformation complete")
    print(f"Rows          : {len(df):,}")
    print(f"Output        : {output_file}")


def transform_products():
    input_file = PROCESSED_DIR / "olist_products_dataset.csv"
    output_file = WAREHOUSE_DIR / "products_clean.csv"

    df = pd.read_csv(input_file)

    df = clean_column_names(df)
    df = df.drop_duplicates()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Products transformation complete")
    print(f"Rows          : {len(df):,}")
    print(f"Output        : {output_file}")


def transform_payments():
    input_file = PROCESSED_DIR / "olist_order_payments_dataset.csv"
    output_file = WAREHOUSE_DIR / "payments_clean.csv"

    df = pd.read_csv(input_file)

    df = clean_column_names(df)
    df = df.drop_duplicates()

    for column in ["payment_installments", "payment_value"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Payments transformation complete")
    print(f"Rows          : {len(df):,}")
    print(f"Output        : {output_file}")


def main():
    transform_orders()
    transform_order_items()
    transform_customers()
    transform_products()
    transform_payments()

    print("\nAll transformations completed successfully.")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd


WAREHOUSE_DIR = Path("data/warehouse")


def load_table(filename):
    return pd.read_csv(WAREHOUSE_DIR / filename)


def check_foreign_key(
    fact_df,
    fact_column,
    dimension_df,
    dimension_column,
    relationship_name,
):
    fact_values = set(
        fact_df[fact_column].dropna()
    )

    dimension_values = set(
        dimension_df[dimension_column].dropna()
    )

    invalid_values = fact_values - dimension_values

    print(f"\n{relationship_name}")
    print("-" * 70)
    print(f"Fact unique keys: {len(fact_values):,}")
    print(f"Dimension unique keys: {len(dimension_values):,}")
    print(f"Invalid keys: {len(invalid_values):,}")

    if invalid_values:
        print("Status: FAILED")
        print(
            "Sample invalid keys:",
            list(invalid_values)[:10],
        )
    else:
        print("Status: PASSED")


def check_primary_key(df, column, table_name):
    total_rows = len(df)
    unique_values = df[column].nunique()
    duplicate_count = total_rows - unique_values

    print(f"\n{table_name}.{column}")
    print("-" * 70)
    print(f"Rows: {total_rows:,}")
    print(f"Unique keys: {unique_values:,}")
    print(f"Duplicate keys: {duplicate_count:,}")

    if duplicate_count == 0:
        print("Status: PASSED")
    else:
        print("Status: FAILED")


def main():

    print("=" * 70)
    print("WAREHOUSE VALIDATION")
    print("=" * 70)

    dim_customer = load_table(
        "dim_customer.csv"
    )

    dim_product = load_table(
        "dim_product.csv"
    )

    dim_seller = load_table(
        "dim_seller.csv"
    )

    dim_date = load_table(
        "dim_date.csv"
    )

    fact_sales = load_table(
        "fact_sales.csv"
    )

    fact_payments = load_table(
        "fact_payments.csv"
    )

    fact_reviews = load_table(
        "fact_reviews.csv"
    )

    # --------------------------------------------------------
    # Fact sales foreign-key validation
    # --------------------------------------------------------

    check_foreign_key(
        fact_sales,
        "customer_key",
        dim_customer,
        "customer_key",
        "fact_sales.customer_key -> dim_customer.customer_key",
    )

    check_foreign_key(
        fact_sales,
        "product_key",
        dim_product,
        "product_key",
        "fact_sales.product_key -> dim_product.product_key",
    )

    check_foreign_key(
        fact_sales,
        "seller_key",
        dim_seller,
        "seller_key",
        "fact_sales.seller_key -> dim_seller.seller_key",
    )

    check_foreign_key(
        fact_sales,
        "order_date_key",
        dim_date,
        "date_key",
        "fact_sales.order_date_key -> dim_date.date_key",
    )

    # --------------------------------------------------------
    # Primary-key validation
    # --------------------------------------------------------

    check_primary_key(
        dim_customer,
        "customer_key",
        "dim_customer",
    )

    check_primary_key(
        dim_product,
        "product_key",
        "dim_product",
    )

    check_primary_key(
        dim_seller,
        "seller_key",
        "dim_seller",
    )

    check_primary_key(
        dim_date,
        "date_key",
        "dim_date",
    )

    check_primary_key(
        fact_sales,
        "sales_key",
        "fact_sales",
    )

    check_primary_key(
        fact_payments,
        "payment_key",
        "fact_payments",
    )

    check_primary_key(
        fact_reviews,
        "review_key",
        "fact_reviews",
    )

    print("\n" + "=" * 70)
    print("WAREHOUSE VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def load_csv(filename):
    return pd.read_csv(RAW_DIR / filename)


def check_relationship(child_df, child_column, parent_df, parent_column, relationship_name):
    child_values = set(child_df[child_column].dropna().unique())
    parent_values = set(parent_df[parent_column].dropna().unique())

    orphan_values = child_values - parent_values

    print(f"\n{relationship_name}")
    print("-" * 70)
    print(f"Child rows: {len(child_df):,}")
    print(f"Unique child keys: {len(child_values):,}")
    print(f"Unique parent keys: {len(parent_values):,}")
    print(f"Orphan keys: {len(orphan_values):,}")

    if orphan_values:
        print("Status: FAILED")
        print("Sample orphan keys:", list(orphan_values)[:10])
    else:
        print("Status: PASSED")


def main():
    print("=" * 70)
    print("OLIST DATA RELATIONSHIP VALIDATION")
    print("=" * 70)

    customers = load_csv("olist_customers_dataset.csv")
    orders = load_csv("olist_orders_dataset.csv")
    order_items = load_csv("olist_order_items_dataset.csv")
    products = load_csv("olist_products_dataset.csv")
    sellers = load_csv("olist_sellers_dataset.csv")
    payments = load_csv("olist_order_payments_dataset.csv")
    reviews = load_csv("olist_order_reviews_dataset.csv")
    translation = load_csv("product_category_name_translation.csv")

    check_relationship(
        orders,
        "customer_id",
        customers,
        "customer_id",
        "orders.customer_id -> customers.customer_id",
    )

    check_relationship(
        order_items,
        "order_id",
        orders,
        "order_id",
        "order_items.order_id -> orders.order_id",
    )

    check_relationship(
        order_items,
        "product_id",
        products,
        "product_id",
        "order_items.product_id -> products.product_id",
    )

    check_relationship(
        order_items,
        "seller_id",
        sellers,
        "seller_id",
        "order_items.seller_id -> sellers.seller_id",
    )

    check_relationship(
        payments,
        "order_id",
        orders,
        "order_id",
        "order_payments.order_id -> orders.order_id",
    )

    check_relationship(
        reviews,
        "order_id",
        orders,
        "order_id",
        "order_reviews.order_id -> orders.order_id",
    )

    check_relationship(
        translation,
        "product_category_name",
        products,
        "product_category_name",
        "translation.product_category_name -> products.product_category_name",
    )

    print("\n" + "=" * 70)
    print("RELATIONSHIP VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
from itertools import combinations
from pathlib import Path

import pandas as pd
import sqlite3


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# repo/
# └── src/
#     └── ml/
#         └── data/
#             └── prepare_product_copurchase.py
#
# parents[3] = repository root
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

DB_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "product_copurchase_pairs.csv"
)


def main():
    print("=" * 70)
    print("PREPARING PRODUCT CO-PURCHASE PAIRS")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate database path
    # -----------------------------------------------------------------
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    conn = sqlite3.connect(DB_PATH)

    # -----------------------------------------------------------------
    # One row per order/product combination.
    #
    # DISTINCT prevents the same product from being counted multiple
    # times within the same order.
    # -----------------------------------------------------------------
    query = """
    SELECT DISTINCT
        f.order_id,
        f.product_key,
        p.product_id,
        p.product_category_name_english
    FROM fact_sales f
    INNER JOIN dim_product p
        ON f.product_key = p.product_key
    WHERE f.product_key IS NOT NULL
    """

    items = pd.read_sql_query(
        query,
        conn,
    )

    conn.close()

    print(
        f"Unique order-product rows: "
        f"{len(items):,}"
    )

    print(
        f"Unique orders: "
        f"{items['order_id'].nunique():,}"
    )

    print(
        f"Unique products: "
        f"{items['product_key'].nunique():,}"
    )

    # -----------------------------------------------------------------
    # Count distinct products per order.
    # -----------------------------------------------------------------
    order_product_counts = (
        items
        .groupby("order_id")["product_key"]
        .nunique()
    )

    multi_product_orders = (
        order_product_counts[
            order_product_counts > 1
        ]
    )

    multi_product_order_count = (
        len(multi_product_orders)
    )

    total_order_count = (
        order_product_counts.size
    )

    multi_product_order_rate = (
        multi_product_order_count
        / total_order_count
        * 100
    )

    print(
        f"Orders with multiple products: "
        f"{multi_product_order_count:,}"
    )

    print(
        f"Multi-product order rate: "
        f"{multi_product_order_rate:.2f}%"
    )

    # -----------------------------------------------------------------
    # Product lookup.
    # -----------------------------------------------------------------
    product_lookup = (
        items[
            [
                "product_key",
                "product_id",
                "product_category_name_english",
            ]
        ]
        .drop_duplicates(
            subset=["product_key"]
        )
        .set_index(
            "product_key"
        )
    )

    # -----------------------------------------------------------------
    # Generate unordered product pairs within each order.
    #
    # Example:
    #
    # Order:
    # A, B, C
    #
    # Pairs:
    # A-B
    # A-C
    # B-C
    #
    # Sorting product keys ensures that A-B and B-A are not treated
    # as separate relationships.
    # -----------------------------------------------------------------
    pair_records = []

    for order_id, group in items.groupby(
        "order_id"
    ):
        product_keys = sorted(
            group["product_key"]
            .dropna()
            .unique()
            .tolist()
        )

        if len(product_keys) < 2:
            continue

        for product_a, product_b in combinations(
            product_keys,
            2,
        ):
            pair_records.append(
                {
                    "order_id": order_id,
                    "product_a_key": product_a,
                    "product_b_key": product_b,
                }
            )

    if not pair_records:
        raise ValueError(
            "No product co-purchase pairs were generated."
        )

    pair_orders = pd.DataFrame(
        pair_records
    )

    # -----------------------------------------------------------------
    # Aggregate pair frequency.
    # -----------------------------------------------------------------
    pairs = (
        pair_orders
        .groupby(
            [
                "product_a_key",
                "product_b_key",
            ]
        )
        .agg(
            co_purchase_count=(
                "order_id",
                "nunique",
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Add product details.
    # -----------------------------------------------------------------
    pairs["product_a_id"] = (
        pairs["product_a_key"]
        .map(
            product_lookup["product_id"]
        )
    )

    pairs["product_b_id"] = (
        pairs["product_b_key"]
        .map(
            product_lookup["product_id"]
        )
    )

    pairs["product_a_category"] = (
        pairs["product_a_key"]
        .map(
            product_lookup[
                "product_category_name_english"
            ]
        )
        .fillna("Unknown")
    )

    pairs["product_b_category"] = (
        pairs["product_b_key"]
        .map(
            product_lookup[
                "product_category_name_english"
            ]
        )
        .fillna("Unknown")
    )

    # -----------------------------------------------------------------
    # Sort by strongest relationship.
    # -----------------------------------------------------------------
    pairs = (
        pairs
        .sort_values(
            "co_purchase_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Validation.
    # -----------------------------------------------------------------
    duplicate_pair_count = int(
        pairs.duplicated(
            subset=[
                "product_a_key",
                "product_b_key",
            ]
        ).sum()
    )

    missing_product_ids = int(
        pairs[
            [
                "product_a_id",
                "product_b_id",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    max_copurchase_count = int(
        pairs[
            "co_purchase_count"
        ].max()
    )

    median_copurchase_count = (
        pairs[
            "co_purchase_count"
        ].median()
    )

    pairs_count_ge_2 = int(
        (
            pairs["co_purchase_count"] >= 2
        ).sum()
    )

    pairs_count_ge_3 = int(
        (
            pairs["co_purchase_count"] >= 3
        ).sum()
    )

    print("\nPAIR DATASET SUMMARY")
    print("-" * 70)

    print(
        f"Product pairs: "
        f"{len(pairs):,}"
    )

    print(
        f"Unique Product A: "
        f"{pairs['product_a_key'].nunique():,}"
    )

    print(
        f"Unique Product B: "
        f"{pairs['product_b_key'].nunique():,}"
    )

    print(
        f"Maximum co-purchase count: "
        f"{max_copurchase_count:,}"
    )

    print(
        f"Median co-purchase count: "
        f"{median_copurchase_count:.0f}"
    )

    print(
        f"Pairs with count >= 2: "
        f"{pairs_count_ge_2:,}"
    )

    print(
        f"Pairs with count >= 3: "
        f"{pairs_count_ge_3:,}"
    )

    print("\nTOP 20 CO-PURCHASE PAIRS")
    print("-" * 70)

    display_columns = [
        "product_a_id",
        "product_b_id",
        "product_a_category",
        "product_b_category",
        "co_purchase_count",
    ]

    print(
        pairs[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print(
        f"\nMissing product IDs in pairs: "
        f"{missing_product_ids}"
    )

    print(
        f"Duplicate pair rows: "
        f"{duplicate_pair_count}"
    )

    # -----------------------------------------------------------------
    # Save output.
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
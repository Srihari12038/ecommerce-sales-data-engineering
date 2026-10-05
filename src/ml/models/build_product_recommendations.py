from pathlib import Path

import pandas as pd
import sqlite3


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

DB_PATH = (
    BASE_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

PAIR_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "product_copurchase_pairs.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "product_recommendations.csv"
)


# ---------------------------------------------------------------------
# Minimum evidence threshold.
#
# Pairs observed only once are too weak for a recommendation rule.
# ---------------------------------------------------------------------
MIN_COPURCHASE_COUNT = 2


def main():
    print("=" * 70)
    print("BUILDING PRODUCT ASSOCIATION RECOMMENDATIONS")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------
    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Co-purchase file not found:\n{PAIR_PATH}"
        )

    # -----------------------------------------------------------------
    # Load co-purchase pairs
    # -----------------------------------------------------------------
    pairs = pd.read_csv(
        PAIR_PATH
    )

    pairs = pairs[
        pairs["co_purchase_count"]
        >= MIN_COPURCHASE_COUNT
    ].copy()

    if pairs.empty:
        raise ValueError(
            "No product pairs remain after applying "
            f"minimum co-purchase count = "
            f"{MIN_COPURCHASE_COUNT}."
        )

    # -----------------------------------------------------------------
    # Get product-level order frequency.
    # -----------------------------------------------------------------
    conn = sqlite3.connect(DB_PATH)

    product_frequency_query = """
    SELECT
        f.product_key,
        COUNT(DISTINCT f.order_id) AS order_count
    FROM fact_sales f
    WHERE f.product_key IS NOT NULL
    GROUP BY f.product_key
    """

    product_frequency = pd.read_sql_query(
        product_frequency_query,
        conn,
    )

    total_orders_query = """
    SELECT COUNT(DISTINCT order_id)
    FROM fact_sales
    """

    total_orders = int(
        conn.execute(
            total_orders_query
        ).fetchone()[0]
    )

    conn.close()

    # -----------------------------------------------------------------
    # Product A frequency
    # -----------------------------------------------------------------
    pairs = pairs.merge(
        product_frequency.rename(
            columns={
                "product_key": "product_a_key",
                "order_count": "product_a_order_count",
            }
        ),
        on="product_a_key",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------------------
    # Product B frequency
    # -----------------------------------------------------------------
    pairs = pairs.merge(
        product_frequency.rename(
            columns={
                "product_key": "product_b_key",
                "order_count": "product_b_order_count",
            }
        ),
        on="product_b_key",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------------------
    # Validate frequency joins
    # -----------------------------------------------------------------
    if (
        pairs[
            [
                "product_a_order_count",
                "product_b_order_count",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Product frequency information is missing "
            "for one or more pairs."
        )

    # -----------------------------------------------------------------
    # Confidence:
    #
    # P(B | A)
    # = co-purchase(A,B) / orders containing A
    #
    # P(A | B)
    # = co-purchase(A,B) / orders containing B
    # -----------------------------------------------------------------
    pairs["confidence_a_to_b"] = (
        pairs["co_purchase_count"]
        / pairs["product_a_order_count"]
    )

    pairs["confidence_b_to_a"] = (
        pairs["co_purchase_count"]
        / pairs["product_b_order_count"]
    )

    # -----------------------------------------------------------------
    # Product support
    # -----------------------------------------------------------------
    pairs["support_a"] = (
        pairs["product_a_order_count"]
        / total_orders
    )

    pairs["support_b"] = (
        pairs["product_b_order_count"]
        / total_orders
    )

    pairs["pair_support"] = (
        pairs["co_purchase_count"]
        / total_orders
    )

    # -----------------------------------------------------------------
    # Lift:
    #
    # lift(A,B)
    # = P(A and B) / (P(A) * P(B))
    # -----------------------------------------------------------------
    pairs["lift"] = (
        pairs["pair_support"]
        / (
            pairs["support_a"]
            * pairs["support_b"]
        )
    )

    # -----------------------------------------------------------------
    # Symmetric confidence.
    # Useful for ranking an item-to-item relationship.
    # -----------------------------------------------------------------
    pairs["symmetric_confidence"] = (
        (
            pairs["confidence_a_to_b"]
            + pairs["confidence_b_to_a"]
        )
        / 2
    )

    # -----------------------------------------------------------------
    # Recommendation score.
    #
    # Lift measures association strength.
    # Symmetric confidence measures practical co-occurrence.
    #
    # We use a simple combined score rather than treating lift alone
    # as sufficient.
    # -----------------------------------------------------------------
    pairs["recommendation_score"] = (
        pairs["lift"]
        * pairs["symmetric_confidence"]
    )

    # -----------------------------------------------------------------
    # Create both recommendation directions:
    #
    # A -> B
    # B -> A
    # -----------------------------------------------------------------
    forward = pairs[
        [
            "product_a_key",
            "product_b_key",
            "product_a_id",
            "product_b_id",
            "product_a_category",
            "product_b_category",
            "co_purchase_count",
            "product_a_order_count",
            "product_b_order_count",
            "confidence_a_to_b",
            "confidence_b_to_a",
            "support_a",
            "support_b",
            "pair_support",
            "lift",
            "symmetric_confidence",
            "recommendation_score",
        ]
    ].copy()

    forward = forward.rename(
        columns={
            "product_a_key": "source_product_key",
            "product_b_key": "recommended_product_key",
            "product_a_id": "source_product_id",
            "product_b_id": "recommended_product_id",
            "product_a_category": "source_category",
            "product_b_category": "recommended_category",
            "product_a_order_count": "source_order_count",
            "product_b_order_count": "recommended_order_count",
            "confidence_a_to_b": "directional_confidence",
        }
    )

    forward = forward.drop(
        columns=[
            "confidence_b_to_a",
            "support_a",
            "support_b",
        ]
    )

    reverse = pairs[
        [
            "product_a_key",
            "product_b_key",
            "product_a_id",
            "product_b_id",
            "product_a_category",
            "product_b_category",
            "co_purchase_count",
            "product_a_order_count",
            "product_b_order_count",
            "confidence_a_to_b",
            "confidence_b_to_a",
            "support_a",
            "support_b",
            "pair_support",
            "lift",
            "symmetric_confidence",
            "recommendation_score",
        ]
    ].copy()

    reverse = reverse.rename(
        columns={
            "product_a_key": "recommended_product_key",
            "product_b_key": "source_product_key",
            "product_a_id": "recommended_product_id",
            "product_b_id": "source_product_id",
            "product_a_category": "recommended_category",
            "product_b_category": "source_category",
            "product_a_order_count": "recommended_order_count",
            "product_b_order_count": "source_order_count",
            "confidence_b_to_a": "directional_confidence",
        }
    )

    reverse = reverse.drop(
        columns=[
            "confidence_a_to_b",
            "support_a",
            "support_b",
        ]
    )

    recommendations = pd.concat(
        [
            forward,
            reverse,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Rank recommendations within each source product.
    # -----------------------------------------------------------------
    recommendations = (
        recommendations
        .sort_values(
            [
                "source_product_key",
                "recommendation_score",
                "co_purchase_count",
                "lift",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    recommendations["recommendation_rank"] = (
        recommendations
        .groupby("source_product_key")
        .cumcount()
        + 1
    )

    # -----------------------------------------------------------------
    # Keep top 10 recommendations per source product.
    # -----------------------------------------------------------------
    recommendations = recommendations[
        recommendations["recommendation_rank"]
        <= 10
    ].copy()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    duplicate_pairs = int(
        recommendations.duplicated(
            subset=[
                "source_product_key",
                "recommended_product_key",
            ]
        ).sum()
    )

    recommendations_per_product = (
        recommendations
        .groupby("source_product_key")
        .size()
    )

    print("\nRECOMMENDATION SUMMARY")
    print("-" * 70)

    print(
        f"Input pairs with >= "
        f"{MIN_COPURCHASE_COUNT} co-purchases: "
        f"{len(pairs):,}"
    )

    print(
        f"Directional recommendation rules: "
        f"{len(recommendations):,}"
    )

    print(
        f"Source products with recommendations: "
        f"{recommendations['source_product_key'].nunique():,}"
    )

    print(
        f"Average recommendations per source: "
        f"{recommendations_per_product.mean():.2f}"
    )

    print(
        f"Duplicate directional pairs: "
        f"{duplicate_pairs}"
    )

    print(
        f"Maximum recommendation rank: "
        f"{recommendations['recommendation_rank'].max()}"
    )

    print("\nTOP RECOMMENDATION RULES")
    print("-" * 70)

    print(
        recommendations[
            [
                "source_product_id",
                "recommended_product_id",
                "source_category",
                "recommended_category",
                "co_purchase_count",
                "directional_confidence",
                "lift",
                "recommendation_score",
                "recommendation_rank",
            ]
        ]
        .sort_values(
            [
                "recommendation_score",
                "co_purchase_count",
            ],
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
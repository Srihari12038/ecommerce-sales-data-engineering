from itertools import combinations
from pathlib import Path

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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "product_recommender_evaluation.csv"
)


# ---------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------
TRAIN_RATIO = 0.80

MIN_COPURCHASE_LEVELS = [
    2,
    3,
    5,
]

K_VALUES = [
    3,
    5,
    10,
]


def build_order_product_data():
    """
    Load one distinct product per order and purchase timestamp.
    """
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT DISTINCT
        f.order_id,
        f.product_key,
        o.order_purchase_timestamp
    FROM fact_sales f
    INNER JOIN dim_order o
        ON f.order_id = o.order_id
    WHERE f.product_key IS NOT NULL
      AND o.order_purchase_timestamp IS NOT NULL
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    conn.close()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "order_id",
                "product_key",
                "order_purchase_timestamp",
            ]
        )
        .sort_values(
            "order_purchase_timestamp"
        )
        .reset_index(drop=True)
    )

    return df


def split_orders(df):
    """
    Perform a chronological order-level 80/20 split.
    """
    order_dates = (
        df[
            [
                "order_id",
                "order_purchase_timestamp",
            ]
        ]
        .drop_duplicates("order_id")
        .sort_values(
            "order_purchase_timestamp"
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(order_dates) * TRAIN_RATIO
    )

    train_order_ids = set(
        order_dates.iloc[
            :split_index
        ]["order_id"]
    )

    test_order_ids = set(
        order_dates.iloc[
            split_index:
        ]["order_id"]
    )

    train = df[
        df["order_id"].isin(
            train_order_ids
        )
    ].copy()

    test = df[
        df["order_id"].isin(
            test_order_ids
        )
    ].copy()

    return train, test, order_dates.iloc[
        :split_index
    ], order_dates.iloc[
        split_index:
    ]


def build_training_pairs(train_df):
    """
    Build unordered product co-purchase pairs using training orders only.
    """
    pair_records = []

    for order_id, group in train_df.groupby(
        "order_id"
    ):
        products = sorted(
            group["product_key"]
            .unique()
            .tolist()
        )

        if len(products) < 2:
            continue

        for product_a, product_b in combinations(
            products,
            2,
        ):
            pair_records.append(
                (
                    product_a,
                    product_b,
                    order_id,
                )
            )

    if not pair_records:
        return pd.DataFrame(
            columns=[
                "product_a_key",
                "product_b_key",
                "co_purchase_count",
            ]
        )

    pair_df = pd.DataFrame(
        pair_records,
        columns=[
            "product_a_key",
            "product_b_key",
            "order_id",
        ],
    )

    pair_df = (
        pair_df
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

    return pair_df


def build_recommendation_map(
    pair_df,
    min_copurchase,
):
    """
    Build directional recommendation maps:
        source product -> recommended products

    Ranking uses:
        co_purchase_count
        then product frequency
    """
    filtered = pair_df[
        pair_df["co_purchase_count"]
        >= min_copurchase
    ].copy()

    if filtered.empty:
        return {}

    # Product frequency derived from pair occurrence is not sufficient
    # for all products, so the caller only uses this map for products
    # that actually participate in qualifying pairs.
    recommendation_map = {}

    for _, row in filtered.iterrows():

        product_a = int(
            row["product_a_key"]
        )

        product_b = int(
            row["product_b_key"]
        )

        count = int(
            row["co_purchase_count"]
        )

        recommendation_map.setdefault(
            product_a,
            []
        ).append(
            (
                product_b,
                count,
            )
        )

        recommendation_map.setdefault(
            product_b,
            []
        ).append(
            (
                product_a,
                count,
            )
        )

    # Sort each source's candidates by strongest historical
    # co-purchase count.
    for source_product in recommendation_map:

        recommendation_map[
            source_product
        ] = sorted(
            recommendation_map[
                source_product
            ],
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )

    return recommendation_map


def evaluate_k(
    test_df,
    recommendation_map,
    k,
):
    """
    Evaluate product recommendations from future orders.

    Each product in a multi-product test order acts as a source.
    The remaining products in that same order are considered
    relevant companion products.

    Metrics:
        precision@k
        recall@k
        hit_rate@k
    """
    precisions = []
    recalls = []
    hits = []

    evaluated_source_count = 0
    source_with_recommendations = 0

    # Group test products by order.
    order_products = {
        order_id: set(
            group["product_key"]
        )
        for order_id, group in test_df.groupby(
            "order_id"
        )
    }

    for products in order_products.values():

        if len(products) < 2:
            continue

        for source_product in products:

            relevant_products = (
                products
                - {source_product}
            )

            if not relevant_products:
                continue

            evaluated_source_count += 1

            candidates = recommendation_map.get(
                int(source_product),
                [],
            )

            if candidates:
                source_with_recommendations += 1

            recommended = [
                product_id
                for product_id, _ in candidates[:k]
            ]

            recommended_set = set(
                recommended
            )

            true_positives = len(
                recommended_set
                & relevant_products
            )

            precision = (
                true_positives / k
            )

            recall = (
                true_positives
                / len(relevant_products)
            )

            hit = (
                1
                if true_positives > 0
                else 0
            )

            precisions.append(
                precision
            )

            recalls.append(
                recall
            )

            hits.append(
                hit
            )

    if evaluated_source_count == 0:
        return {
            "evaluated_source_instances": 0,
            "source_coverage": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
        }

    return {
        "evaluated_source_instances": (
            evaluated_source_count
        ),
        "source_coverage": (
            source_with_recommendations
            / evaluated_source_count
        ),
        "precision_at_k": (
            sum(precisions)
            / len(precisions)
        ),
        "recall_at_k": (
            sum(recalls)
            / len(recalls)
        ),
        "hit_rate_at_k": (
            sum(hits)
            / len(hits)
        ),
    }


def main():
    print("=" * 70)
    print("TIME-AWARE PRODUCT RECOMMENDER EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    df = build_order_product_data()

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Orders:              "
        f"{df['order_id'].nunique():,}"
    )

    print(
        f"Products:            "
        f"{df['product_key'].nunique():,}"
    )

    print(
        f"Order-product rows:  "
        f"{len(df):,}"
    )

    # -----------------------------------------------------------------
    # Chronological split
    # -----------------------------------------------------------------
    train, test, train_orders, test_orders = (
        split_orders(df)
    )

    print("\nTIME SPLIT")
    print("-" * 70)

    print(
        f"Training orders: "
        f"{len(train_orders):,}"
    )

    print(
        f"Testing orders:  "
        f"{len(test_orders):,}"
    )

    print(
        f"Training period: "
        f"{train_orders['order_purchase_timestamp'].min()} "
        f"to "
        f"{train_orders['order_purchase_timestamp'].max()}"
    )

    print(
        f"Testing period:  "
        f"{test_orders['order_purchase_timestamp'].min()} "
        f"to "
        f"{test_orders['order_purchase_timestamp'].max()}"
    )

    # -----------------------------------------------------------------
    # Multi-product test orders
    # -----------------------------------------------------------------
    test_order_product_counts = (
        test
        .groupby("order_id")["product_key"]
        .nunique()
    )

    multi_product_test_orders = int(
        (
            test_order_product_counts > 1
        ).sum()
    )

    print(
        f"\nTest orders with multiple products: "
        f"{multi_product_test_orders:,}"
    )

    # -----------------------------------------------------------------
    # Build training associations.
    # -----------------------------------------------------------------
    print("\nBUILDING TRAINING ASSOCIATIONS...")
    print("-" * 70)

    pair_df = build_training_pairs(
        train
    )

    print(
        f"Training product pairs: "
        f"{len(pair_df):,}"
    )

    # -----------------------------------------------------------------
    # Evaluate every contamination/evidence level and K.
    # -----------------------------------------------------------------
    results = []

    for min_copurchase in (
        MIN_COPURCHASE_LEVELS
    ):

        recommendation_map = (
            build_recommendation_map(
                pair_df,
                min_copurchase,
            )
        )

        qualifying_pairs = int(
            (
                pair_df[
                    "co_purchase_count"
                ]
                >= min_copurchase
            ).sum()
        )

        source_products = len(
            recommendation_map
        )

        print(
            f"\nMinimum co-purchases = "
            f"{min_copurchase}"
        )

        print(
            f"Qualifying pairs: "
            f"{qualifying_pairs:,}"
        )

        print(
            f"Source products: "
            f"{source_products:,}"
        )

        for k in K_VALUES:

            metrics = evaluate_k(
                test,
                recommendation_map,
                k,
            )

            result = {
                "min_copurchase": (
                    min_copurchase
                ),
                "k": k,
                "training_pairs": len(
                    pair_df
                ),
                "qualifying_pairs": (
                    qualifying_pairs
                ),
                "source_products": (
                    source_products
                ),
                **metrics,
            }

            results.append(
                result
            )

            print(
                f"K={k:<2} "
                f"Precision={metrics['precision_at_k']:.4f} "
                f"Recall={metrics['recall_at_k']:.4f} "
                f"HitRate={metrics['hit_rate_at_k']:.4f} "
                f"Coverage={metrics['source_coverage']:.4f}"
            )

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------------------
    # Select operating point.
    #
    # Primary:
    #   hit rate
    #
    # Secondary:
    #   precision
    #   source coverage
    #
    # A very small evidence threshold that performs only because
    # of extremely rare pairs should not automatically win.
    # -----------------------------------------------------------------
    candidate_results = (
        results_df
        .sort_values(
            [
                "hit_rate_at_k",
                "precision_at_k",
                "source_coverage",
                "min_copurchase",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    if candidate_results.empty:
        raise ValueError(
            "No recommender evaluation results were generated."
        )

    best = candidate_results.iloc[
        0
    ]

    results_df["selection_status"] = (
        "Not Selected"
    )

    selection_mask = (
        (results_df["min_copurchase"]
         == best["min_copurchase"])
        &
        (results_df["k"]
         == best["k"])
    )

    results_df.loc[
        selection_mask,
        "selection_status",
    ] = "Selected"

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("\nFINAL EVALUATION")
    print("-" * 70)

    print(
        results_df[
            [
                "min_copurchase",
                "k",
                "qualifying_pairs",
                "source_products",
                "precision_at_k",
                "recall_at_k",
                "hit_rate_at_k",
                "source_coverage",
                "selection_status",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSELECTED OPERATING POINT")
    print("-" * 70)

    print(
        f"Minimum co-purchases: "
        f"{int(best['min_copurchase'])}"
    )

    print(
        f"K: "
        f"{int(best['k'])}"
    )

    print(
        f"Precision@K: "
        f"{best['precision_at_k']:.4f}"
    )

    print(
        f"Recall@K: "
        f"{best['recall_at_k']:.4f}"
    )

    print(
        f"Hit Rate@K: "
        f"{best['hit_rate_at_k']:.4f}"
    )

    print(
        f"Source coverage: "
        f"{best['source_coverage']:.4f}"
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DB_PATH = BASE_DIR / "data" / "database" / "ecommerce.db"
EVALUATION_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "product_recommender_evaluation.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "product_recommender_report.csv"
)


# ============================================================
# LOAD EVALUATION
# ============================================================

evaluation = pd.read_csv(EVALUATION_PATH)

if evaluation.empty:
    raise ValueError("Recommendation evaluation file is empty.")


# ============================================================
# SELECT FINAL OPERATING POINT
# ============================================================

selected = evaluation[
    evaluation["selection_status"].astype(str).str.lower() == "selected"
].copy()

if selected.empty:
    raise ValueError("No selected operating point found.")

selected_row = selected.iloc[0]


# ============================================================
# DATASET STATISTICS
# ============================================================

conn = sqlite3.connect(DB_PATH)

orders_query = """
SELECT
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT product_key) AS total_products
FROM fact_sales
WHERE product_key IS NOT NULL
"""

orders_result = pd.read_sql_query(orders_query, conn).iloc[0]

multi_order_query = """
SELECT
    COUNT(*) AS multi_product_orders
FROM (
    SELECT
        order_id,
        COUNT(DISTINCT product_key) AS product_count
    FROM fact_sales
    WHERE product_key IS NOT NULL
    GROUP BY order_id
    HAVING COUNT(DISTINCT product_key) > 1
)
"""

multi_product_orders = int(
    pd.read_sql_query(multi_order_query, conn).iloc[0]["multi_product_orders"]
)

conn.close()


total_orders = int(orders_result["total_orders"])
total_products = int(orders_result["total_products"])

multi_product_percentage = (
    multi_product_orders / total_orders * 100
    if total_orders > 0
    else 0
)


# ============================================================
# INTERPRETATION
# ============================================================

precision = float(selected_row["precision_at_k"])
recall = float(selected_row["recall_at_k"])
hit_rate = float(selected_row["hit_rate_at_k"])
coverage = float(selected_row["source_coverage"])

if hit_rate < 0.05:
    performance = "Limited predictive power"
elif hit_rate < 0.15:
    performance = "Moderate predictive power"
else:
    performance = "Strong predictive power"


conclusion = (
    "The item-to-item recommendation model was retained as a baseline "
    "association-based recommendation prototype. The selected operating "
    "point uses a minimum of 2 historical co-purchases and generates the "
    "top 5 recommendations. Predictive performance is limited because "
    "only a small proportion of orders contain multiple distinct products, "
    "resulting in sparse co-purchase relationships. The model should "
    "therefore be presented as a recommendation prototype rather than a "
    "high-performing production recommender."
)


# ============================================================
# CREATE REPORT
# ============================================================

report = pd.DataFrame(
    [
        {
            "module": "Product Recommendation",
            "method": "Item-to-Item Co-Purchase Association",
            "evaluation_method": "Chronological 80/20 Order-Level Split",
            "total_orders": total_orders,
            "total_products": total_products,
            "multi_product_orders": multi_product_orders,
            "multi_product_order_percentage": round(
                multi_product_percentage, 4
            ),
            "minimum_copurchases": int(
                selected_row["min_copurchase"]
            ),
            "k": int(selected_row["k"]),
            "precision_at_k": round(precision, 6),
            "recall_at_k": round(recall, 6),
            "hit_rate_at_k": round(hit_rate, 6),
            "source_coverage": round(coverage, 6),
            "performance_assessment": performance,
            "selection_status": "Selected",
            "conclusion": conclusion,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

report.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("PRODUCT RECOMMENDATION FINAL REPORT")
print("=" * 70)

print("\nDATASET")
print("-" * 70)
print(f"Total orders:                  {total_orders:,}")
print(f"Total products:                {total_products:,}")
print(f"Multi-product orders:          {multi_product_orders:,}")
print(
    f"Multi-product order percentage: {multi_product_percentage:.2f}%"
)

print("\nSELECTED OPERATING POINT")
print("-" * 70)
print(
    f"Minimum co-purchases:          "
    f"{int(selected_row['min_copurchase'])}"
)
print(f"K:                              {int(selected_row['k'])}")

print("\nPERFORMANCE")
print("-" * 70)
print(f"Precision@K:                   {precision:.4f}")
print(f"Recall@K:                      {recall:.4f}")
print(f"Hit Rate@K:                    {hit_rate:.4f}")
print(f"Source Coverage:               {coverage:.4f}")

print("\nASSESSMENT")
print("-" * 70)
print(performance)

print("\nCONCLUSION")
print("-" * 70)
print(conclusion)

print("\nSAVED")
print("-" * 70)
print(OUTPUT_PATH)

print("=" * 70)
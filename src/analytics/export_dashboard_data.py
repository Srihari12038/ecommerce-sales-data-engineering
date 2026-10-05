import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "dashboard"
    / "data"
)


QUERIES = {
    "executive_kpi": """
        SELECT
            ROUND(SUM(price), 2) AS total_sales,
            ROUND(SUM(freight_value), 2) AS total_freight,
            COUNT(*) AS total_items,
            COUNT(DISTINCT order_id) AS total_orders,
            COUNT(DISTINCT customer_key) AS active_customers,
            ROUND(
                SUM(price) / COUNT(DISTINCT order_id),
                2
            ) AS average_order_value
        FROM fact_sales;
    """,

    "monthly_sales": """
        SELECT
            d.year,
            d.month,
            d.month_name,
            ROUND(SUM(f.price), 2) AS total_sales,
            ROUND(SUM(f.freight_value), 2) AS total_freight,
            COUNT(*) AS total_items,
            COUNT(DISTINCT f.order_id) AS total_orders
        FROM fact_sales f
        JOIN dim_date d
            ON f.order_date_key = d.date_key
        GROUP BY
            d.year,
            d.month,
            d.month_name
        ORDER BY
            d.year,
            d.month;
    """,

    "category_sales": """
        SELECT
            COALESCE(
                p.product_category_name_english,
                'Unknown'
            ) AS product_category,
            ROUND(SUM(f.price), 2) AS total_sales,
            ROUND(SUM(f.freight_value), 2) AS total_freight,
            COUNT(*) AS total_items,
            COUNT(DISTINCT f.order_id) AS total_orders
        FROM fact_sales f
        JOIN dim_product p
            ON f.product_key = p.product_key
        GROUP BY
            p.product_category_name_english
        ORDER BY
            total_sales DESC;
    """,

    "state_sales": """
        SELECT
            c.customer_state AS state,
            ROUND(SUM(f.price), 2) AS total_sales,
            ROUND(SUM(f.freight_value), 2) AS total_freight,
            COUNT(DISTINCT f.order_id) AS total_orders,
            COUNT(*) AS total_items
        FROM fact_sales f
        JOIN dim_customer c
            ON f.customer_key = c.customer_key
        GROUP BY
            c.customer_state
        ORDER BY
            total_sales DESC;
    """,

    "payment_analysis": """
        SELECT
            payment_type,
            COUNT(*) AS total_transactions,
            ROUND(SUM(payment_value), 2) AS total_payment_value,
            ROUND(AVG(payment_value), 2) AS average_payment_value,
            ROUND(AVG(payment_installments), 2)
                AS average_installments
        FROM fact_payments
        GROUP BY
            payment_type
        ORDER BY
            total_payment_value DESC;
    """,

    "review_analysis": """
        SELECT
            review_score,
            COUNT(*) AS total_reviews,
            ROUND(
                COUNT(*) * 100.0 /
                (SELECT COUNT(*) FROM fact_reviews),
                2
            ) AS percentage_of_reviews
        FROM fact_reviews
        GROUP BY
            review_score
        ORDER BY
            review_score;
    """,

    "order_status": """
        SELECT
            order_status,
            COUNT(*) AS total_orders,
            ROUND(
                COUNT(*) * 100.0 /
                (SELECT COUNT(*) FROM dim_order),
                2
            ) AS percentage_of_orders
        FROM dim_order
        GROUP BY
            order_status
        ORDER BY
            total_orders DESC;
    """,

    "top_products": """
        SELECT
            p.product_id,
            p.product_category_name_english AS product_category,
            ROUND(SUM(f.price), 2) AS total_sales,
            ROUND(SUM(f.freight_value), 2) AS total_freight,
            COUNT(*) AS total_items,
            COUNT(DISTINCT f.order_id) AS total_orders
        FROM fact_sales f
        JOIN dim_product p
            ON f.product_key = p.product_key
        GROUP BY
            p.product_id,
            p.product_category_name_english
        ORDER BY
            total_sales DESC
        LIMIT 10;
    """
}


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    try:
        for name, query in QUERIES.items():

            dataframe = pd.read_sql_query(
                query,
                connection
            )

            output_file = (
                OUTPUT_DIR
                / f"{name}.csv"
            )

            dataframe.to_csv(
                output_file,
                index=False
            )

            print(
                f"Exported: {output_file} "
                f"({len(dataframe)} rows)"
            )

    finally:
        connection.close()

    print("=" * 70)
    print("DASHBOARD DATA EXPORT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
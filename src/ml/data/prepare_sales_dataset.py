import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATABASE_FILE = (
    PROJECT_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "monthly_sales_ml.csv"
)


def main():
    connection = sqlite3.connect(DATABASE_FILE)

    query = """
    SELECT
        d.year,
        d.month,
        d.month_name,

        SUM(f.price) AS total_sales,

        SUM(f.freight_value) AS total_freight,

        SUM(f.total_item_value) AS total_item_value,

        COUNT(*) AS total_items,

        COUNT(DISTINCT f.order_id) AS total_orders,

        COUNT(DISTINCT f.customer_key) AS active_customers

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
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-01"
    )

    df = df[
        [
            "date",
            "year",
            "month",
            "month_name",
            "total_sales",
            "total_freight",
            "total_item_value",
            "total_items",
            "total_orders",
            "active_customers",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 80)
    print("ML SALES DATASET CREATED")
    print("=" * 80)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("=" * 80)


if __name__ == "__main__":
    main()
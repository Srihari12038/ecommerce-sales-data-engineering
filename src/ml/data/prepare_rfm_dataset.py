import sqlite3
import pandas as pd

from pathlib import Path


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
    / "customer_rfm.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Connect to warehouse
    # ---------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # ---------------------------------------------------------
    # 2. RFM query
    # ---------------------------------------------------------

    query = """
    SELECT
        f.customer_key,
        MAX(d.full_date) AS last_purchase_date,
        COUNT(DISTINCT f.order_id) AS frequency,
        SUM(f.price) AS monetary
    FROM fact_sales f
    JOIN dim_date d
        ON f.order_date_key = d.date_key
    GROUP BY f.customer_key
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    # ---------------------------------------------------------
    # 3. Convert date
    # ---------------------------------------------------------

    df["last_purchase_date"] = pd.to_datetime(
        df["last_purchase_date"]
    )

    # ---------------------------------------------------------
    # 4. Determine analysis date
    # ---------------------------------------------------------

    analysis_date = (
        df["last_purchase_date"].max()
        + pd.Timedelta(days=1)
    )

    # ---------------------------------------------------------
    # 5. Calculate Recency
    # ---------------------------------------------------------

    df["recency"] = (
        analysis_date
        - df["last_purchase_date"]
    ).dt.days

    # ---------------------------------------------------------
    # 6. Rename monetary metric
    # ---------------------------------------------------------

    df["monetary"] = (
        df["monetary"]
        .round(2)
    )

    # ---------------------------------------------------------
    # 7. Select final columns
    # ---------------------------------------------------------

    rfm = df[
        [
            "customer_key",
            "last_purchase_date",
            "recency",
            "frequency",
            "monetary"
        ]
    ].copy()

    # ---------------------------------------------------------
    # 8. Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rfm.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 9. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("CUSTOMER RFM DATASET")
    print("=" * 80)

    print(f"Customers: {len(rfm)}")

    print(
        f"Analysis date: "
        f"{analysis_date.date()}"
    )

    print("\nRFM Statistics")
    print("-" * 80)

    print(
        rfm[
            [
                "recency",
                "frequency",
                "monetary"
            ]
        ].describe().to_string()
    )

    print("\nTop Customers by Monetary Value")
    print("-" * 80)

    print(
        rfm
        .sort_values(
            "monetary",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
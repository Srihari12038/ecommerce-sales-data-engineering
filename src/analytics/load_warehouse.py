from pathlib import Path
import sqlite3

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

WAREHOUSE_DIR = PROJECT_DIR / "data" / "warehouse"

DATABASE_DIR = PROJECT_DIR / "data" / "database"

DATABASE_FILE = DATABASE_DIR / "ecommerce.db"


# ============================================================
# TABLES
# ============================================================

TABLES = {
    "dim_customer": "dim_customer.csv",
    "dim_product": "dim_product.csv",
    "dim_seller": "dim_seller.csv",
    "dim_date": "dim_date.csv",
    "dim_order": "dim_order.csv",
    "fact_sales": "fact_sales.csv",
    "fact_payments": "fact_payments.csv",
    "fact_reviews": "fact_reviews.csv",
}


# ============================================================
# LOAD TABLE
# ============================================================

def load_table(connection, table_name, filename):
    file_path = WAREHOUSE_DIR / filename

    print(f"Loading {filename}...")

    df = pd.read_csv(file_path)

    df.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False,
    )

    print(
        f"Loaded {table_name}: {len(df):,} rows"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOADING STAR SCHEMA INTO SQLITE")
    print("=" * 70)

    # Create database directory
    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Connect to SQLite
    connection = sqlite3.connect(
        DATABASE_FILE
    )

    try:

        for table_name, filename in TABLES.items():

            load_table(
                connection,
                table_name,
                filename,
            )

        connection.commit()

    finally:

        connection.close()

    print("\n" + "=" * 70)
    print("SQLITE WAREHOUSE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nDatabase:")
    print(DATABASE_FILE)


if __name__ == "__main__":
    main()
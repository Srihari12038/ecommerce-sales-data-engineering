import sqlite3
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_DIR
    / "data"
    / "database"
    / "ecommerce.db"
)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python src/analytics/run_sql.py <sql_file>"
        )
        return

    sql_file = Path(sys.argv[1])

    if not sql_file.exists():
        print(f"SQL file not found: {sql_file}")
        return

    sql = sql_file.read_text(
        encoding="utf-8"
    ).strip()

    if not sql:
        print(f"SQL file is empty: {sql_file}")
        return

    connection = sqlite3.connect(DATABASE_FILE)

    try:
        cursor = connection.execute(sql)

        if cursor.description is None:
            print("SQL executed successfully.")
            print("This query did not return a result set.")
            return

        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchall()

        print("=" * 80)
        print(f"SQL QUERY: {sql_file}")
        print("=" * 80)

        print(" | ".join(columns))

        print("-" * 80)

        for row in rows:
            print(" | ".join(str(value) for value in row))

        print("-" * 80)
        print(f"Rows returned: {len(rows)}")

        print("=" * 80)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
import sqlite3
from pathlib import Path


DATABASE_FILE = Path("data/database/ecommerce.db")


def main():
    connection = sqlite3.connect(DATABASE_FILE)

    try:
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """

        tables = connection.execute(query).fetchall()

        print("=" * 60)
        print("SQLITE DATABASE TABLES")
        print("=" * 60)

        for table in tables:
            print(table[0])

        print("=" * 60)
        print(f"Total tables: {len(tables)}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
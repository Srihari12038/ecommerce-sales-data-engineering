import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_csv(file_path):
    """Load a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def ingest_data():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw")

    print(f"Found {len(csv_files)} CSV files.\n")

    for file_path in csv_files:
        print(f"Reading: {file_path.name}")

        df = load_csv(file_path)

        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {len(df.columns)}")

        output_path = PROCESSED_DIR / file_path.name
        df.to_csv(output_path, index=False)

        print(f"  Saved   : {output_path}\n")


if __name__ == "__main__":
    ingest_data()

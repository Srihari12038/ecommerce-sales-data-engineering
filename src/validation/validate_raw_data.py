import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def validate_file(file_path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"Rows        : {len(df):,}")
    print(f"Columns     : {len(df.columns)}")
    print(f"Duplicates  : {df.duplicated().sum():,}")

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            print(f"  {column}: {count:,} ({percentage:.2f}%)")

    print("\nData types:")
    print(df.dtypes.to_string())

def main():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        print("ERROR: No CSV files found in data/raw/")
        return

    print(f"Found {len(files)} CSV files.")

    for file_path in files:
        try:
            validate_file(file_path)
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")

if __name__ == "__main__":
    main()

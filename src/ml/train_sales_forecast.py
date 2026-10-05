import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Load feature dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("=" * 80)
    print("TIME-AWARE SALES FORECAST DATASET")
    print("=" * 80)

    print(f"Total observations: {len(df)}")

    # ---------------------------------------------------------
    # 2. Define features
    # ---------------------------------------------------------

    features = [
        "lag_1_sales",
        "lag_2_sales",
        "lag_3_sales",
        "rolling_3_sales",
        "rolling_6_sales",
        "lag_1_sales_growth",
        "lag_1_orders",
        "lag_1_items",
        "lag_1_customers",
        "quarter",
        "month_sin",
        "month_cos"
    ]

    target = "total_sales"

    # ---------------------------------------------------------
    # 3. Time-based split
    # ---------------------------------------------------------

    test_size = 5

    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # ---------------------------------------------------------
    # 4. Display split information
    # ---------------------------------------------------------

    print("\nTRAINING DATA")
    print("-" * 80)

    print(
        f"Rows: {len(train)}"
    )

    print(
        f"Period: "
        f"{train['date'].min().date()} "
        f"to "
        f"{train['date'].max().date()}"
    )

    print("\nTEST DATA")
    print("-" * 80)

    print(
        f"Rows: {len(test)}"
    )

    print(
        f"Period: "
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    # ---------------------------------------------------------
    # 5. Display feature matrix
    # ---------------------------------------------------------

    print("\nFEATURES")
    print("-" * 80)

    for feature in features:
        print(f"- {feature}")

    print("\nTARGET")
    print("-" * 80)

    print(f"- {target}")

    # ---------------------------------------------------------
    # 6. Verify chronological ordering
    # ---------------------------------------------------------

    chronological = (
        train["date"].max()
        <
        test["date"].min()
    )

    print("\nTIME ORDER VALIDATION")
    print("-" * 80)

    print(
        f"Training ends before testing starts: "
        f"{chronological}"
    )

    # ---------------------------------------------------------
    # 7. Preview test data
    # ---------------------------------------------------------

    print("\nTEST DATA PREVIEW")
    print("-" * 80)

    print(
        test[
            [
                "date",
                "total_sales",
                "lag_1_sales",
                "rolling_3_sales"
            ]
        ]
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
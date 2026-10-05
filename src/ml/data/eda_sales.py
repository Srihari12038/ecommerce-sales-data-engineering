import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "monthly_sales_ml.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "eda"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print("=" * 80)
    print("ML SALES DATASET - EDA")
    print("=" * 80)

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDate range:")
    print(df["date"].min())
    print(df["date"].max())

    print("\nSales statistics:")
    print(df["total_sales"].describe())

    print("\nMonthly dataset:")
    print(
        df[
            [
                "date",
                "total_sales",
                "total_orders",
                "total_items",
                "active_customers"
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Check missing calendar months
    # ---------------------------------------------------------

    complete_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS"
    )

    missing_months = complete_dates[
        ~complete_dates.isin(df["date"])
    ]

    print("\nMissing calendar months:")

    if len(missing_months) == 0:
        print("None")
    else:
        for date in missing_months:
            print(date.strftime("%Y-%m"))

    # ---------------------------------------------------------
    # Sales trend
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["total_sales"],
        marker="o"
    )

    plt.title("Monthly E-Commerce Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    sales_chart = OUTPUT_DIR / "monthly_sales_trend.png"

    plt.savefig(
        sales_chart,
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Orders trend
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["total_orders"],
        marker="o"
    )

    plt.title("Monthly Orders")
    plt.xlabel("Date")
    plt.ylabel("Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()

    orders_chart = OUTPUT_DIR / "monthly_orders_trend.png"

    plt.savefig(
        orders_chart,
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    numeric_columns = [
        "total_sales",
        "total_freight",
        "total_item_value",
        "total_items",
        "total_orders",
        "active_customers"
    ]

    correlation = df[numeric_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation.round(3))

    print("\nEDA completed successfully.")

    print(f"\nEDA output directory:")
    print(OUTPUT_DIR)

    print("=" * 80)


if __name__ == "__main__":
    main()
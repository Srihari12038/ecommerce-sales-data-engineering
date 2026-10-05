import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "monthly_sales_ml.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Create complete monthly calendar
    # ---------------------------------------------------------

    full_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS"
    )

    df = (
        df.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )

    # Restore calendar fields
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")

    # Missing months represent no observed sales
    numeric_columns = [
        "total_sales",
        "total_freight",
        "total_item_value",
        "total_items",
        "total_orders",
        "active_customers"
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(0)

    # ---------------------------------------------------------
    # 3. Lag features
    # ---------------------------------------------------------

    df["lag_1_sales"] = (
        df["total_sales"].shift(1)
    )

    df["lag_2_sales"] = (
        df["total_sales"].shift(2)
    )

    df["lag_3_sales"] = (
        df["total_sales"].shift(3)
    )

    # ---------------------------------------------------------
    # 4. Rolling features
    # ---------------------------------------------------------

    df["rolling_3_sales"] = (
        df["total_sales"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["rolling_6_sales"] = (
        df["total_sales"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    # ---------------------------------------------------------
    # 5. Lagged growth
    # ---------------------------------------------------------

    sales_growth = (
        df["total_sales"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Shift so current sales are never used
    # to predict current sales.
    df["lag_1_sales_growth"] = (
        sales_growth.shift(1)
    )

    # ---------------------------------------------------------
    # 6. Lagged business metrics
    # ---------------------------------------------------------

    df["lag_1_orders"] = (
        df["total_orders"].shift(1)
    )

    df["lag_1_items"] = (
        df["total_items"].shift(1)
    )

    df["lag_1_customers"] = (
        df["active_customers"].shift(1)
    )

    # ---------------------------------------------------------
    # 7. Calendar features
    # ---------------------------------------------------------

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["month_sin"] = (
        np.sin(
            2 * np.pi * df["month"] / 12
        )
    )

    df["month_cos"] = (
        np.cos(
            2 * np.pi * df["month"] / 12
        )
    )

    # ---------------------------------------------------------
    # 8. Remove rows without sufficient history
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[
            "lag_1_sales",
            "lag_2_sales",
            "lag_3_sales",
            "rolling_3_sales",
            "rolling_6_sales"
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 9. Save feature dataset
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 10. Report
    # ---------------------------------------------------------

    print("=" * 80)
    print("SALES FEATURE ENGINEERING COMPLETED")
    print("=" * 80)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFeature columns:")

    feature_columns = [
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

    for column in feature_columns:
        print(f"- {column}")

    print("\nDate range:")
    print(
        df["date"].min(),
        "to",
        df["date"].max()
    )

    print("\nPreview:")
    print(
        df[
            [
                "date",
                "total_sales",
                "lag_1_sales",
                "lag_2_sales",
                "lag_3_sales",
                "rolling_3_sales",
                "rolling_6_sales"
            ]
        ].tail(10)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
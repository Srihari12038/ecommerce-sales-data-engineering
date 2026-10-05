import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)


def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    # Avoid division by zero
    non_zero = actual != 0

    if np.any(non_zero):
        mape = np.mean(
            np.abs(
                (
                    actual[non_zero]
                    - predicted[non_zero]
                )
                / actual[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def main():

    # ---------------------------------------------------------
    # 1. Load dataset
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

    # ---------------------------------------------------------
    # 2. Time-based test split
    # ---------------------------------------------------------

    test_size = 5

    test = df.iloc[-test_size:].copy()

    # ---------------------------------------------------------
    # 3. Naive forecast
    # ---------------------------------------------------------

    # Prediction for the current month is
    # the previous month's actual sales.

    test["baseline_prediction"] = (
        test["lag_1_sales"]
    )

    # ---------------------------------------------------------
    # 4. Calculate metrics
    # ---------------------------------------------------------

    mae, rmse, mape = calculate_metrics(
        test["total_sales"],
        test["baseline_prediction"]
    )

    # ---------------------------------------------------------
    # 5. Display results
    # ---------------------------------------------------------

    print("=" * 80)
    print("NAIVE SALES FORECAST BASELINE")
    print("=" * 80)

    print("\nForecast Method:")
    print("Previous month's sales")

    print("\nTest Period:")
    print(
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    print("\nBASELINE METRICS")
    print("-" * 80)

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")

    print("\nACTUAL VS BASELINE")
    print("-" * 80)

    result = test[
        [
            "date",
            "total_sales",
            "baseline_prediction"
        ]
    ].copy()

    result["error"] = (
        result["total_sales"]
        - result["baseline_prediction"]
    )

    print(result.to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()
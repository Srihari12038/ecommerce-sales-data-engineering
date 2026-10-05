import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "predictions"
    / "final_sales_forecast.csv"
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

    # ---------------------------------------------------------
    # 2. Generate one-step-ahead forecast
    # ---------------------------------------------------------

    # Selected forecasting strategy:
    # Naive Baseline
    #
    # Forecast = previous month's actual sales

    df["forecast_sales"] = (
        df["lag_1_sales"]
    )

    # ---------------------------------------------------------
    # 3. Forecast error
    # ---------------------------------------------------------

    df["forecast_error"] = (
        df["total_sales"]
        - df["forecast_sales"]
    )

    df["absolute_error"] = (
        df["forecast_error"]
        .abs()
    )

    # ---------------------------------------------------------
    # 4. Forecast status
    # ---------------------------------------------------------

    df["forecast_status"] = "Historical"

    # ---------------------------------------------------------
    # 5. Select output columns
    # ---------------------------------------------------------

    forecast = df[
        [
            "date",
            "total_sales",
            "forecast_sales",
            "forecast_error",
            "absolute_error",
            "forecast_status"
        ]
    ].copy()

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("FINAL SALES FORECAST PIPELINE")
    print("=" * 80)

    print("\nSelected Model:")
    print("Naive Baseline")

    print("\nForecast Method:")
    print("Previous month's actual sales")

    print("\nRows:")
    print(len(forecast))

    print("\nForecast Preview:")
    print(
        forecast.tail(10).to_string(
            index=False
        )
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
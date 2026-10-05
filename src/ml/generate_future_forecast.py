import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

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
    / "predictions"
    / "future_sales_forecast.csv"
)


FORECAST_MONTHS = 6


def main():

    # ---------------------------------------------------------
    # 1. Load historical monthly sales
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
    # 2. Get latest historical observation
    # ---------------------------------------------------------

    latest_date = df["date"].max()

    latest_sales = (
        df.loc[
            df["date"] == latest_date,
            "total_sales"
        ].iloc[0]
    )

    # ---------------------------------------------------------
    # 3. Generate future dates
    # ---------------------------------------------------------

    future_dates = pd.date_range(
        start=latest_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS"
    )

    # ---------------------------------------------------------
    # 4. Generate naive forecasts
    # ---------------------------------------------------------

    # Selected model:
    #
    # Forecast = previous month's actual sales
    #
    # For recursive multi-step forecasting, the previous
    # forecast becomes the next month's input.

    forecasts = []

    previous_sales = latest_sales

    for forecast_date in future_dates:

        forecast_sales = previous_sales

        forecasts.append(
            {
                "forecast_date": forecast_date,
                "forecast_sales": forecast_sales,
                "forecast_method": "Naive Baseline",
                "forecast_status": "Future"
            }
        )

        previous_sales = forecast_sales

    # ---------------------------------------------------------
    # 5. Create output dataframe
    # ---------------------------------------------------------

    forecast_df = pd.DataFrame(
        forecasts
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    print("=" * 80)
    print("FUTURE SALES FORECAST")
    print("=" * 80)

    print("\nHistorical Data Through:")
    print(latest_date.date())

    print("\nLatest Historical Sales:")
    print(f"₹{latest_sales:,.2f}")

    print("\nForecast Horizon:")
    print(f"{FORECAST_MONTHS} months")

    print("\nFUTURE FORECAST")
    print("-" * 80)

    print(
        forecast_df.to_string(
            index=False
        )
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
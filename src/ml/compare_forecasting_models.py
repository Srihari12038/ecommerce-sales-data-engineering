import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)

RF_PREDICTION_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "predictions"
    / "random_forest_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "model_comparison.csv"
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

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = denominator != 0

    smape = np.mean(
        2
        * np.abs(
            actual[valid] - predicted[valid]
        )
        / denominator[valid]
    ) * 100

    return mae, rmse, smape


def main():

    # ---------------------------------------------------------
    # 1. Load feature dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    test_size = 5

    test = df.iloc[-test_size:].copy()

    # ---------------------------------------------------------
    # 2. Naive baseline
    # ---------------------------------------------------------

    baseline_predictions = (
        test["lag_1_sales"]
    )

    baseline_mae, baseline_rmse, baseline_smape = (
        calculate_metrics(
            test["total_sales"],
            baseline_predictions
        )
    )

    # ---------------------------------------------------------
    # 3. Random Forest predictions
    # ---------------------------------------------------------

    rf = pd.read_csv(
        RF_PREDICTION_FILE,
        parse_dates=["date"]
    )

    rf_mae, rf_rmse, rf_smape = (
        calculate_metrics(
            rf["actual_sales"],
            rf["predicted_sales"]
        )
    )

    # ---------------------------------------------------------
    # 4. Build comparison table
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "Naive Baseline",
                "mae": baseline_mae,
                "rmse": baseline_rmse,
                "smape": baseline_smape
            },
            {
                "model": "Random Forest",
                "mae": rf_mae,
                "rmse": rf_rmse,
                "smape": rf_smape
            }
        ]
    )

    # ---------------------------------------------------------
    # 5. Calculate baseline improvement
    # ---------------------------------------------------------

    baseline_mae_value = baseline_mae

    comparison["mae_change_vs_baseline"] = (
        (
            comparison["mae"]
            - baseline_mae_value
        )
        / baseline_mae_value
    ) * 100

    # ---------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("FORECASTING MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nINTERPRETATION")
    print("-" * 80)

    best_model = comparison.loc[
        comparison["mae"].idxmin(),
        "model"
    ]

    print(
        f"Best model based on MAE: "
        f"{best_model}"
    )

    print(
        "\nNegative MAE change = improvement "
        "over baseline."
    )

    print(
        "Positive MAE change = worse than baseline."
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
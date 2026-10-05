import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


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
    / "evaluation"
    / "walk_forward_results.csv"
)


FEATURES = [
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

TARGET = "total_sales"


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
    # 1. Load data
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
    print("WALK-FORWARD TIME-SERIES VALIDATION")
    print("=" * 80)

    print(f"Total observations: {len(df)}")

    # ---------------------------------------------------------
    # 2. Minimum training size
    # ---------------------------------------------------------

    minimum_train_size = 10

    results = []

    # ---------------------------------------------------------
    # 3. Walk-forward validation
    # ---------------------------------------------------------

    for test_index in range(
        minimum_train_size,
        len(df)
    ):

        train = df.iloc[:test_index]
        test = df.iloc[test_index:test_index + 1]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        actual = y_test.iloc[0]

        # -----------------------------------------------------
        # Naive baseline
        # -----------------------------------------------------

        naive_prediction = (
            test["lag_1_sales"].iloc[0]
        )

        # -----------------------------------------------------
        # Random Forest
        # -----------------------------------------------------

        random_forest = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        random_forest.fit(
            X_train,
            y_train
        )

        rf_prediction = (
            random_forest.predict(X_test)[0]
        )

        # -----------------------------------------------------
        # Gradient Boosting
        # -----------------------------------------------------

        gradient_boosting = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42
        )

        gradient_boosting.fit(
            X_train,
            y_train
        )

        gb_prediction = (
            gradient_boosting.predict(X_test)[0]
        )

        # -----------------------------------------------------
        # Save predictions
        # -----------------------------------------------------

        predictions = {
            "Naive Baseline": naive_prediction,
            "Random Forest": rf_prediction,
            "Gradient Boosting": gb_prediction
        }

        for model_name, prediction in predictions.items():

            mae, rmse, smape = calculate_metrics(
                [actual],
                [prediction]
            )

            results.append(
                {
                    "date": test["date"].iloc[0],
                    "model": model_name,
                    "actual_sales": actual,
                    "predicted_sales": prediction,
                    "mae": mae,
                    "rmse": rmse,
                    "smape": smape
                }
            )

    # ---------------------------------------------------------
    # 4. Results dataframe
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 5. Average model performance
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("model")
        .agg(
            average_mae=("mae", "mean"),
            average_rmse=("rmse", "mean"),
            average_smape=("smape", "mean")
        )
        .reset_index()
        .sort_values("average_mae")
    )

    # ---------------------------------------------------------
    # 6. Save detailed results
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display validation results
    # ---------------------------------------------------------

    print("\nWALK-FORWARD PREDICTIONS")
    print("-" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nAVERAGE MODEL PERFORMANCE")
    print("-" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 8. Best model
    # ---------------------------------------------------------

    best_model = summary.iloc[0]["model"]

    print("\nMODEL SELECTION")
    print("-" * 80)

    print(
        f"Best model based on average MAE: "
        f"{best_model}"
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "sales_features.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "src"
    / "ml"
    / "models"
    / "sales_gradient_boosting.pkl"
)

PREDICTION_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "predictions"
    / "gradient_boosting_predictions.csv"
)


def calculate_smape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = denominator != 0

    return (
        np.mean(
            2
            * np.abs(
                actual[valid] - predicted[valid]
            )
            / denominator[valid]
        )
        * 100
    )


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

    # ---------------------------------------------------------
    # 2. Features and target
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
    # 4. Gradient Boosting model
    # ---------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 5. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 6. Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    smape = calculate_smape(
        y_test,
        predictions
    )

    # ---------------------------------------------------------
    # 7. Results
    # ---------------------------------------------------------

    result = pd.DataFrame(
        {
            "date": test["date"],
            "actual_sales": y_test.values,
            "predicted_sales": predictions
        }
    )

    result["error"] = (
        result["actual_sales"]
        - result["predicted_sales"]
    )

    result["absolute_error"] = (
        np.abs(result["error"])
    )

    # ---------------------------------------------------------
    # 8. Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    # ---------------------------------------------------------
    # 9. Save model
    # ---------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ---------------------------------------------------------
    # 10. Save predictions
    # ---------------------------------------------------------

    PREDICTION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        PREDICTION_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 11. Display results
    # ---------------------------------------------------------

    print("=" * 80)
    print("GRADIENT BOOSTING SALES FORECAST")
    print("=" * 80)

    print("\nMODEL")
    print("-" * 80)

    print("Algorithm: Gradient Boosting Regressor")
    print("Estimators: 100")
    print("Learning Rate: 0.05")
    print("Max Depth: 2")
    print("Minimum Samples Leaf: 2")

    print("\nDATASET")
    print("-" * 80)

    print(f"Training rows: {len(train)}")
    print(f"Testing rows : {len(test)}")

    print("\nMODEL PERFORMANCE")
    print("-" * 80)

    print(f"MAE  : ₹{mae:,.2f}")
    print(f"RMSE : ₹{rmse:,.2f}")
    print(f"sMAPE: {smape:.2f}%")

    print("\nACTUAL VS PREDICTED")
    print("-" * 80)

    print(
        result.to_string(
            index=False
        )
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 80)

    print(
        importance.to_string(
            index=False
        )
    )

    print("\nMODEL FILE")
    print("-" * 80)

    print(MODEL_FILE)

    print("\nPREDICTION FILE")
    print("-" * 80)

    print(PREDICTION_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
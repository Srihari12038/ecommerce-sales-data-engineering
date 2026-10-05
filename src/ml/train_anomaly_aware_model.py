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
    / "sales_features_anomaly_aware.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "src"
    / "ml"
    / "models"
    / "sales_anomaly_aware_gradient_boosting.pkl"
)

PREDICTION_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "predictions"
    / "anomaly_aware_predictions.csv"
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
    "month_cos",
    "anomaly_flag"
]

TARGET = "total_sales"


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
    # 1. Load anomaly-aware dataset
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
    # 2. Time-aware train/test split
    # ---------------------------------------------------------

    test_size = 5

    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # ---------------------------------------------------------
    # 3. Train Gradient Boosting
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
    # 4. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 5. Evaluation
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
    # 6. Prediction results
    # ---------------------------------------------------------

    result = pd.DataFrame(
        {
            "date": test["date"],
            "actual_sales": y_test.values,
            "predicted_sales": predictions,
            "anomaly_flag": test["anomaly_flag"].values
        }
    )

    result["absolute_error"] = (
        np.abs(
            result["actual_sales"]
            - result["predicted_sales"]
        )
    )

    # ---------------------------------------------------------
    # 7. Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    # ---------------------------------------------------------
    # 8. Save model
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
    # 9. Save predictions
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
    # 10. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("ANOMALY-AWARE GRADIENT BOOSTING FORECAST")
    print("=" * 80)

    print("\nFeatures:")
    print("- Anomaly flag included")

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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "ml" / "delivery_features_train.csv"
TEST_PATH = BASE_DIR / "data" / "ml" / "delivery_features_test.csv"

MODEL_DIR = BASE_DIR / "src" / "ml" / "models"
PREDICTION_DIR = BASE_DIR / "data" / "ml" / "predictions"
EVALUATION_DIR = BASE_DIR / "data" / "ml" / "evaluation"

MODEL_PATH = MODEL_DIR / "delivery_gradient_boosting.pkl"
PREDICTION_PATH = (
    PREDICTION_DIR / "delivery_gradient_boosting_predictions.csv"
)
EVALUATION_PATH = (
    EVALUATION_DIR / "delivery_gradient_boosting_results.csv"
)
IMPORTANCE_PATH = (
    EVALUATION_DIR / "delivery_gradient_boosting_feature_importance.csv"
)


def symmetric_mape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    return (
        np.mean(
            2
            * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
        * 100
    )


def main():
    print("=" * 70)
    print("GRADIENT BOOSTING DELIVERY-TIME REGRESSION")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "delivery_days"

    excluded_columns = [
        "order_id",
        "delivery_days",
        "late_delivery_flag",
    ]

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    X_test = test_df[feature_columns]
    y_test = test_df[target]

    print("\nDATA")
    print("-" * 70)
    print(f"Training rows:  {len(X_train):,}")
    print(f"Testing rows:   {len(X_test):,}")
    print(f"Feature count:  {len(feature_columns)}")

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        loss="huber",
    )

    print("\nTraining Gradient Boosting...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    smape = symmetric_mape(
        y_test.to_numpy(),
        predictions,
    )

    print("\nMODEL PERFORMANCE")
    print("-" * 70)
    print(f"MAE:   {mae:.4f} days")
    print(f"RMSE:  {rmse:.4f} days")
    print(f"sMAPE: {smape:.2f}%")

    # ---------------------------------------------------------------
    # Baseline comparison
    # ---------------------------------------------------------------
    median_baseline = np.full(
        len(y_test),
        y_train.median(),
    )

    baseline_mae = mean_absolute_error(
        y_test,
        median_baseline,
    )

    improvement = (
        (baseline_mae - mae)
        / baseline_mae
        * 100
    )

    print("\nBASELINE COMPARISON")
    print("-" * 70)
    print(
        f"Median baseline MAE: {baseline_mae:.4f} days"
    )
    print(
        f"GB MAE improvement:   {improvement:.2f}%"
    )

    if mae < baseline_mae:
        print(
            "Result: Gradient Boosting BEATS "
            "the median baseline."
        )
    else:
        print(
            "Result: Gradient Boosting DOES NOT "
            "beat the median baseline."
        )

    # ---------------------------------------------------------------
    # Compare with Random Forest if its result exists.
    # ---------------------------------------------------------------
    rf_results_path = (
        EVALUATION_DIR
        / "delivery_random_forest_results.csv"
    )

    if rf_results_path.exists():
        rf_results = pd.read_csv(rf_results_path)

        if not rf_results.empty:
            rf_mae = rf_results.loc[0, "mae"]

            print("\nRANDOM FOREST COMPARISON")
            print("-" * 70)
            print(f"Random Forest MAE: {rf_mae:.4f} days")
            print(f"Gradient Boosting:  {mae:.4f} days")

            if mae < rf_mae:
                print(
                    "Result: Gradient Boosting "
                    "BEATS Random Forest."
                )
            else:
                print(
                    "Result: Random Forest "
                    "REMAINS BETTER."
                )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nTOP 15 FEATURES")
    print("-" * 70)

    for _, row in importance_df.head(15).iterrows():
        print(
            f"{row['feature']:<35} "
            f"{row['importance']:.6f}"
        )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    prediction_df = pd.DataFrame(
        {
            "order_id": test_df["order_id"],
            "actual_delivery_days": y_test,
            "predicted_delivery_days": predictions,
            "absolute_error_days": np.abs(
                y_test.to_numpy() - predictions
            ),
        }
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    evaluation_df = pd.DataFrame(
        [
            {
                "problem": "delivery_time_regression",
                "model": "gradient_boosting",
                "mae": mae,
                "rmse": rmse,
                "smape": smape,
                "baseline_mae": baseline_mae,
                "mae_improvement_pct": improvement,
            }
        ]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nSAVED")
    print("-" * 70)
    print(MODEL_PATH)
    print(PREDICTION_PATH)
    print(EVALUATION_PATH)
    print(IMPORTANCE_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
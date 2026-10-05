from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "delivery_features_train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "delivery_features_test.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "ml" / "evaluation"
OUTPUT_PATH = OUTPUT_DIR / "delivery_baseline_results.csv"


def symmetric_mape(y_true, y_pred):
    denominator = (
        np.abs(y_true) + np.abs(y_pred)
    )

    mask = denominator != 0

    return (
        np.mean(
            2
            * np.abs(
                y_pred[mask] - y_true[mask]
            )
            / denominator[mask]
        )
        * 100
    )


def main():
    print("=" * 70)
    print("DELIVERY PREDICTION BASELINES")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # ================================================================
    # REGRESSION BASELINE
    # ================================================================

    y_train_reg = train_df["delivery_days"]
    y_test_reg = test_df["delivery_days"]

    median_prediction = y_train_reg.median()
    mean_prediction = y_train_reg.mean()

    median_pred = np.full(
        len(y_test_reg),
        median_prediction,
    )

    mean_pred = np.full(
        len(y_test_reg),
        mean_prediction,
    )

    print("\nREGRESSION BASELINE")
    print("-" * 70)

    print(
        f"Training median delivery: "
        f"{median_prediction:.2f} days"
    )

    print(
        f"Training mean delivery:   "
        f"{mean_prediction:.2f} days"
    )

    median_mae = mean_absolute_error(
        y_test_reg,
        median_pred,
    )

    median_rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            median_pred,
        )
    )

    median_smape = symmetric_mape(
        y_test_reg.to_numpy(),
        median_pred,
    )

    mean_mae = mean_absolute_error(
        y_test_reg,
        mean_pred,
    )

    mean_rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            mean_pred,
        )
    )

    mean_smape = symmetric_mape(
        y_test_reg.to_numpy(),
        mean_pred,
    )

    print("\nMedian baseline:")
    print(f"  MAE:   {median_mae:.4f} days")
    print(f"  RMSE:  {median_rmse:.4f} days")
    print(f"  sMAPE: {median_smape:.2f}%")

    print("\nMean baseline:")
    print(f"  MAE:   {mean_mae:.4f} days")
    print(f"  RMSE:  {mean_rmse:.4f} days")
    print(f"  sMAPE: {mean_smape:.2f}%")

    # ================================================================
    # CLASSIFICATION BASELINE
    # ================================================================

    y_test_cls = test_df["late_delivery_flag"]

    majority_prediction = 0

    cls_pred = np.full(
        len(y_test_cls),
        majority_prediction,
    )

    accuracy = accuracy_score(
        y_test_cls,
        cls_pred,
    )

    precision = precision_score(
        y_test_cls,
        cls_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test_cls,
        cls_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test_cls,
        cls_pred,
        zero_division=0,
    )

    print("\nCLASSIFICATION BASELINE")
    print("-" * 70)

    print("Baseline prediction: NOT LATE (0)")

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    results = pd.DataFrame(
        [
            {
                "problem": "delivery_time_regression",
                "model": "median_baseline",
                "mae": median_mae,
                "rmse": median_rmse,
                "smape": median_smape,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            },
            {
                "problem": "delivery_time_regression",
                "model": "mean_baseline",
                "mae": mean_mae,
                "rmse": mean_rmse,
                "smape": mean_smape,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            },
            {
                "problem": "late_delivery_classification",
                "model": "majority_baseline",
                "mae": np.nan,
                "rmse": np.nan,
                "smape": np.nan,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
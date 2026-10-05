from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
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

MODEL_PATH = (
    BASE_DIR
    / "src"
    / "ml"
    / "models"
    / "delivery_gradient_boosting.pkl"
)

PREDICTION_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "predictions"
    / "final_delivery_predictions.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "delivery_model_report.csv"
)


def main():
    print("=" * 70)
    print("FINAL DELIVERY PREDICTIONS")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # The feature datasets intentionally exclude purchase_timestamp. 
    # Retrieve it from the original order-level delivery dataset for
    # output/reporting only. It is NOT used as a model feature.
    SOURCE_PATH = (
        BASE_DIR
        / "data"
        / "ml"
        / "delivery_prediction_dataset.csv"
        )
    
    source_df = pd.read_csv(
        SOURCE_PATH,
        usecols=[
            "order_id",
            "purchase_timestamp",
            ],
            )
    
    test_df = test_df.merge(
        source_df,
        on="order_id",
        how="left",
        validate="one_to_one",
        )

    model = joblib.load(MODEL_PATH)

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
    
    X_test = test_df[feature_columns]
    
    actual_delivery_days = test_df["delivery_days"].to_numpy()
    
    estimated_delivery_days = (
        test_df["estimated_delivery_days"].to_numpy()
        )

    actual_late = (
        test_df["late_delivery_flag"].to_numpy()
        )

    # ---------------------------------------------------------------
    # Delivery-time prediction
    # ---------------------------------------------------------------
    predicted_delivery_days = model.predict(
        X_test
    )

    absolute_error_days = np.abs(
        actual_delivery_days
        - predicted_delivery_days
    )

    # ---------------------------------------------------------------
    # Regression-derived delivery risk
    #
    # Positive value means predicted delivery takes longer
    # than the estimated delivery commitment.
    # ---------------------------------------------------------------
    delivery_risk_score = (
        predicted_delivery_days
        - estimated_delivery_days
    )

    predicted_late = (
        delivery_risk_score > 0
    ).astype(int)

    # ---------------------------------------------------------------
    # Regression metrics
    # ---------------------------------------------------------------
    regression_mae = mean_absolute_error(
        actual_delivery_days,
        predicted_delivery_days,
    )

    regression_rmse = np.sqrt(
        mean_squared_error(
            actual_delivery_days,
            predicted_delivery_days,
        )
    )

    # ---------------------------------------------------------------
    # Risk classification metrics
    # ---------------------------------------------------------------
    risk_accuracy = accuracy_score(
        actual_late,
        predicted_late,
    )

    risk_precision = precision_score(
        actual_late,
        predicted_late,
        zero_division=0,
    )

    risk_recall = recall_score(
        actual_late,
        predicted_late,
        zero_division=0,
    )

    risk_f1 = f1_score(
        actual_late,
        predicted_late,
        zero_division=0,
    )

    risk_roc_auc = roc_auc_score(
        actual_late,
        delivery_risk_score,
    )

    risk_pr_auc = average_precision_score(
        actual_late,
        delivery_risk_score,
    )

    # ---------------------------------------------------------------
    # Risk bands
    #
    # These are operational categories, not probability estimates.
    # ---------------------------------------------------------------
    risk_band = np.select(
        [
            delivery_risk_score <= 0,
            delivery_risk_score <= 2,
            delivery_risk_score <= 5,
        ],
        [
            "On Time Risk",
            "Low Late Risk",
            "Moderate Late Risk",
        ],
        default="High Late Risk",
    )

    # ---------------------------------------------------------------
    # Final prediction dataset
    # ---------------------------------------------------------------
    prediction_df = pd.DataFrame(
        {
            "order_id": test_df["order_id"],
            "purchase_timestamp": test_df[
                "purchase_timestamp"
            ],
            "estimated_delivery_days": estimated_delivery_days,
            "actual_delivery_days": actual_delivery_days,
            "predicted_delivery_days": predicted_delivery_days,
            "absolute_error_days": absolute_error_days,
            "delivery_risk_score": delivery_risk_score,
            "actual_late_delivery": actual_late,
            "predicted_late_delivery": predicted_late,
            "risk_band": risk_band,
        }
    )

    # ---------------------------------------------------------------
    # Formal evaluation report
    # ---------------------------------------------------------------
    report = pd.DataFrame(
        [
            {
                "problem": "delivery_time_regression",
                "model": "gradient_boosting",
                "status": "Selected",
                "mae": regression_mae,
                "rmse": regression_rmse,
                "smape": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            },
            {
                "problem": "late_delivery_risk",
                "model": "regression_derived_risk",
                "status": "Selected",
                "mae": np.nan,
                "rmse": np.nan,
                "smape": np.nan,
                "precision": risk_precision,
                "recall": risk_recall,
                "f1": risk_f1,
                "roc_auc": risk_roc_auc,
                "pr_auc": risk_pr_auc,
            },
        ]
    )

    PREDICTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------
    print("\nDELIVERY-TIME REGRESSION")
    print("-" * 70)
    print(f"MAE:  {regression_mae:.4f} days")
    print(f"RMSE: {regression_rmse:.4f} days")

    print("\nREGRESSION-DERIVED LATE RISK")
    print("-" * 70)
    print(f"Accuracy:  {risk_accuracy:.4f}")
    print(f"Precision: {risk_precision:.4f}")
    print(f"Recall:    {risk_recall:.4f}")
    print(f"F1:        {risk_f1:.4f}")
    print(f"ROC-AUC:   {risk_roc_auc:.4f}")
    print(f"PR-AUC:    {risk_pr_auc:.4f}")

    print("\nBUSINESS OUTPUT")
    print("-" * 70)
    print(
        f"Actual late rate: "
        f"{actual_late.mean() * 100:.2f}%"
    )

    print(
        f"Predicted late rate: "
        f"{predicted_late.mean() * 100:.2f}%"
    )

    print(
        f"Late orders detected: "
        f"{predicted_late.sum():,}"
    )

    print("\nRISK BAND DISTRIBUTION")
    print("-" * 70)
    print(
        prediction_df["risk_band"]
        .value_counts()
        .to_string()
    )

    print("\nSAVED")
    print("-" * 70)
    print(PREDICTION_PATH)
    print(REPORT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

MODEL_DIR = (
    BASE_DIR
    / "src"
    / "ml"
    / "models"
)

PREDICTION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "predictions"
)

EVALUATION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

MODEL_PATH = (
    MODEL_DIR
    / "late_delivery_random_forest_final.pkl"
)

PREDICTION_PATH = (
    PREDICTION_DIR
    / "late_delivery_final_predictions.csv"
)

EVALUATION_PATH = (
    EVALUATION_DIR
    / "late_delivery_final_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    EVALUATION_DIR
    / "late_delivery_final_feature_importance.csv"
)


SELECTED_THRESHOLD = 0.61


def main():
    print("=" * 70)
    print("FINAL LATE DELIVERY MODEL EVALUATION")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "late_delivery_flag"

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

    print("\nFINAL MODEL")
    print("-" * 70)
    print("Model: Random Forest")
    print(f"Frozen threshold: {SELECTED_THRESHOLD:.2f}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Features:      {len(feature_columns)}")

    # ---------------------------------------------------------------
    # Train final model on ALL training data.
    # ---------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining final Random Forest on all training data...")

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------------
    # Final untouched test evaluation.
    # ---------------------------------------------------------------
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\nFINAL TEST PERFORMANCE")
    print("-" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nCONFUSION MATRIX")
    print("-" * 70)
    print(matrix)

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # Positive prediction rate
    # ---------------------------------------------------------------
    actual_late_rate = y_test.mean()
    predicted_late_rate = predictions.mean()

    print("BUSINESS SIGNAL")
    print("-" * 70)
    print(
        f"Actual late rate:     "
        f"{actual_late_rate * 100:.2f}%"
    )

    print(
        f"Predicted late rate:  "
        f"{predicted_late_rate * 100:.2f}%"
    )

    print(
        f"Late orders detected: "
        f"{predictions.sum():,}"
    )

    print(
        f"Actual late orders:   "
        f"{y_test.sum():,}"
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
            f"{row['feature']:<35}"
            f"{row['importance']:.6f}"
        )

    # ---------------------------------------------------------------
    # Prediction output
    # ---------------------------------------------------------------
    prediction_df = pd.DataFrame(
        {
            "order_id": test_df["order_id"],
            "actual_late_delivery": y_test,
            "late_delivery_probability": probabilities,
            "predicted_late_delivery": predictions,
        }
    )

    # ---------------------------------------------------------------
    # Evaluation output
    # ---------------------------------------------------------------
    evaluation_df = pd.DataFrame(
        [
            {
                "problem": "late_delivery_classification",
                "model": "random_forest_final",
                "threshold": SELECTED_THRESHOLD,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "actual_late_rate": actual_late_rate,
                "predicted_late_rate": predicted_late_rate,
            }
        ]
    )

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(MODEL_PATH)
    print(PREDICTION_PATH)
    print(EVALUATION_PATH)
    print(FEATURE_IMPORTANCE_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
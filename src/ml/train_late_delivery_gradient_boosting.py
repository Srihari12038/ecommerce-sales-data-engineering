from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
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

TRAIN_PATH = BASE_DIR / "data" / "ml" / "delivery_features_train.csv"
TEST_PATH = BASE_DIR / "data" / "ml" / "delivery_features_test.csv"

MODEL_DIR = BASE_DIR / "src" / "ml" / "models"
PREDICTION_DIR = BASE_DIR / "data" / "ml" / "predictions"
EVALUATION_DIR = BASE_DIR / "data" / "ml" / "evaluation"

MODEL_PATH = (
    MODEL_DIR / "late_delivery_gradient_boosting.pkl"
)

PREDICTION_PATH = (
    PREDICTION_DIR
    / "late_delivery_gradient_boosting_predictions.csv"
)

EVALUATION_PATH = (
    EVALUATION_DIR
    / "late_delivery_gradient_boosting_results.csv"
)

IMPORTANCE_PATH = (
    EVALUATION_DIR
    / "late_delivery_gradient_boosting_feature_importance.csv"
)


def main():
    print("=" * 70)
    print("GRADIENT BOOSTING LATE DELIVERY CLASSIFICATION")
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

    print("\nDATA")
    print("-" * 70)
    print(f"Training rows:  {len(X_train):,}")
    print(f"Testing rows:   {len(X_test):,}")
    print(f"Feature count:  {len(feature_columns)}")

    print(
        f"Training late rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Testing late rate:  "
        f"{y_test.mean() * 100:.2f}%"
    )

    # ---------------------------------------------------------------
    # Balanced sample weights.
    #
    # Each class receives inverse-frequency weighting.
    # ---------------------------------------------------------------
    positive_count = (y_train == 1).sum()
    negative_count = (y_train == 0).sum()

    total_count = len(y_train)
    class_count = 2

    weight_negative = (
        total_count
        / (class_count * negative_count)
    )

    weight_positive = (
        total_count
        / (class_count * positive_count)
    )

    sample_weights = np.where(
        y_train == 1,
        weight_positive,
        weight_negative,
    )

    print("\nCLASS WEIGHTS")
    print("-" * 70)
    print(f"On-time weight: {weight_negative:.4f}")
    print(f"Late weight:    {weight_positive:.4f}")

    # ---------------------------------------------------------------
    # Gradient Boosting classifier
    # ---------------------------------------------------------------
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        subsample=0.80,
        random_state=42,
    )

    print("\nTraining Gradient Boosting...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
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

    print("\nMODEL PERFORMANCE")
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

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------
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

    print("TOP 15 FEATURES")
    print("-" * 70)

    for _, row in importance_df.head(15).iterrows():
        print(
            f"{row['feature']:<35}"
            f"{row['importance']:.6f}"
        )

    # ---------------------------------------------------------------
    # Predictions
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
    # Evaluation
    # ---------------------------------------------------------------
    evaluation_df = pd.DataFrame(
        [
            {
                "problem": "late_delivery_classification",
                "model": "gradient_boosting",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        ]
    )

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
        IMPORTANCE_PATH,
        index=False,
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
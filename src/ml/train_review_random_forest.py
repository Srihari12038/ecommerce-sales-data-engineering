from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "review_features_train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "review_features_test.csv"
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
    / "review_random_forest.pkl"
)

PREDICTION_PATH = (
    PREDICTION_DIR
    / "review_random_forest_predictions.csv"
)

EVALUATION_PATH = (
    EVALUATION_DIR
    / "review_random_forest_results.csv"
)

IMPORTANCE_PATH = (
    EVALUATION_DIR
    / "review_random_forest_feature_importance.csv"
)


def main():
    print("=" * 70)
    print("RANDOM FOREST REVIEW SCORE CLASSIFICATION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "review_score"

    excluded_columns = [
        "order_id",
        "review_score",
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
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Feature count: {len(feature_columns)}")

    print("\nTRAIN TARGET DISTRIBUTION")
    print("-" * 70)
    print(y_train.value_counts(normalize=True).sort_index().mul(100).round(2))

    print("\nTEST TARGET DISTRIBUTION")
    print("-" * 70)
    print(y_test.value_counts(normalize=True).sort_index().mul(100).round(2))

    # -----------------------------------------------------------------
    # Random Forest
    #
    # Balanced class weighting is important because scores 1-3 are
    # substantially less frequent than score 5.
    # -----------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------
    predictions = model.predict(X_test)

    # -----------------------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # -----------------------------------------------------------------
    # Ordinal metrics
    #
    # Predicting 4 instead of 5 is less severe than predicting 1
    # instead of 5. These metrics capture that structure.
    # -----------------------------------------------------------------
    score_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    quadratic_kappa = cohen_kappa_score(
        y_test,
        predictions,
        weights="quadratic",
    )

    print("\nMODEL PERFORMANCE")
    print("-" * 70)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Balanced Accuracy:   {balanced_accuracy:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Weighted F1:         {weighted_f1:.4f}")
    print(f"Score MAE:           {score_mae:.4f}")
    print(f"Quadratic Kappa:     {quadratic_kappa:.4f}")

    # -----------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3, 4, 5],
    )

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    print(
        pd.DataFrame(
            matrix,
            index=[
                "Actual 1",
                "Actual 2",
                "Actual 3",
                "Actual 4",
                "Actual 5",
            ],
            columns=[
                "Pred 1",
                "Pred 2",
                "Pred 3",
                "Pred 4",
                "Pred 5",
            ],
        )
    )

    # -----------------------------------------------------------------
    # Classification report
    # -----------------------------------------------------------------
    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            labels=[1, 2, 3, 4, 5],
            digits=4,
            zero_division=0,
        )
    )

    # -----------------------------------------------------------------
    # Feature importance
    # -----------------------------------------------------------------
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

    print("TOP 20 FEATURES")
    print("-" * 70)

    for _, row in importance_df.head(20).iterrows():
        print(
            f"{row['feature']:<40}"
            f"{row['importance']:.6f}"
        )

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------
    prediction_df = pd.DataFrame(
        {
            "order_id": test_df["order_id"],
            "actual_review_score": y_test,
            "predicted_review_score": predictions,
            "absolute_score_error": np.abs(
                y_test.to_numpy()
                - predictions
            ),
        }
    )

    # -----------------------------------------------------------------
    # Evaluation record
    # -----------------------------------------------------------------
    evaluation_df = pd.DataFrame(
        [
            {
                "problem": "review_score_classification",
                "model": "random_forest",
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "score_mae": score_mae,
                "quadratic_kappa": quadratic_kappa,
            }
        ]
    )

    # -----------------------------------------------------------------
    # Create directories
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # -----------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------
    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save evaluation
    # -----------------------------------------------------------------
    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save feature importance
    # -----------------------------------------------------------------
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
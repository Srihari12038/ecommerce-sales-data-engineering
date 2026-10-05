from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


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
    / "review_ordinal_gradient_boosting.pkl"
)

PREDICTION_PATH = (
    PREDICTION_DIR
    / "review_ordinal_predictions.csv"
)

EVALUATION_PATH = (
    EVALUATION_DIR
    / "review_ordinal_results.csv"
)


def main():
    print("=" * 70)
    print("ORDINAL REVIEW SCORE REGRESSION")
    print("=" * 70)

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
    y_train = train_df[target].astype(float)

    X_test = test_df[feature_columns]
    y_test = test_df[target].astype(int)

    print("\nDATA")
    print("-" * 70)
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Feature count: {len(feature_columns)}")

    # -----------------------------------------------------------------
    # Ordinal regression model
    # -----------------------------------------------------------------
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.80,
        loss="huber",
        random_state=42,
    )

    print("\nTraining ordinal Gradient Boosting regressor...")

    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------------------
    # Continuous prediction
    # -----------------------------------------------------------------
    continuous_predictions = model.predict(
        X_test
    )

    # -----------------------------------------------------------------
    # Convert to valid review scores
    # -----------------------------------------------------------------
    predictions = np.rint(
        continuous_predictions
    ).astype(int)

    predictions = np.clip(
        predictions,
        1,
        5,
    )

    # -----------------------------------------------------------------
    # Metrics
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

    score_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    score_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
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
    print(f"Score RMSE:          {score_rmse:.4f}")
    print(f"Quadratic Kappa:     {quadratic_kappa:.4f}")

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

    print("\nTOP 20 FEATURES")
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
            "continuous_predicted_score": continuous_predictions,
            "predicted_review_score": predictions,
            "absolute_score_error": np.abs(
                y_test.to_numpy() - predictions
            ),
        }
    )

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    evaluation_df = pd.DataFrame(
        [
            {
                "problem": "review_score_ordinal_prediction",
                "model": "gradient_boosting_regression",
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "score_mae": score_mae,
                "score_rmse": score_rmse,
                "quadratic_kappa": quadratic_kappa,
            }
        ]
    )

    # -----------------------------------------------------------------
    # Save
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
        EVALUATION_DIR
        / "review_ordinal_feature_importance.csv",
        index=False,
    )

    print("\nSAVED")
    print("-" * 70)
    print(MODEL_PATH)
    print(PREDICTION_PATH)
    print(EVALUATION_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
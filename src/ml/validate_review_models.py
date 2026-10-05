from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "review_features_train.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "review_validation_results.csv"
)


def evaluate_predictions(y_true, predictions):
    return {
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "score_mae": mean_absolute_error(
            y_true,
            predictions,
        ),
        "quadratic_kappa": cohen_kappa_score(
            y_true,
            predictions,
            weights="quadratic",
        ),
    }


def main():
    print("=" * 70)
    print("TIME-AWARE REVIEW MODEL VALIDATION")
    print("=" * 70)

    df = pd.read_csv(TRAIN_PATH)

    target = "review_score"

    excluded_columns = [
        "order_id",
        "review_score",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    # -----------------------------------------------------------------
    # The existing training period is split chronologically:
    #
    # 80% -> model training
    # 20% -> validation
    #
    # The final test period remains completely untouched.
    # -----------------------------------------------------------------
    split_index = int(len(df) * 0.80)

    train_core = df.iloc[:split_index].copy()
    validation = df.iloc[split_index:].copy()

    X_core = train_core[feature_columns]
    y_core = train_core[target].astype(int)

    X_val = validation[feature_columns]
    y_val = validation[target].astype(int)

    print("\nVALIDATION SPLIT")
    print("-" * 70)

    print(
        f"Training-core rows: "
        f"{len(train_core):,}"
    )

    print(
        f"Validation rows:    "
        f"{len(validation):,}"
    )

    print(
        f"Training-core period: "
        f"{train_core['order_id'].iloc[0]}"
        " to "
        f"{train_core['order_id'].iloc[-1]}"
    )

    print(
        f"Validation rows begin at chronological position: "
        f"{split_index:,}"
    )

    # -----------------------------------------------------------------
    # Random Forest classifier
    # -----------------------------------------------------------------
    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(
        X_core,
        y_core,
    )

    rf_pred = rf.predict(
        X_val
    ).astype(int)

    rf_metrics = evaluate_predictions(
        y_val,
        rf_pred,
    )

    print("\nRANDOM FOREST VALIDATION")
    print("-" * 70)
    for key, value in rf_metrics.items():
        print(
            f"{key:<20}: {value:.4f}"
        )

    # -----------------------------------------------------------------
    # Gradient Boosting classifier
    # -----------------------------------------------------------------
    print("\nTraining Gradient Boosting classifier...")

    class_counts = (
        y_core
        .value_counts()
        .sort_index()
    )

    total_samples = len(y_core)
    number_of_classes = len(class_counts)

    class_weights = {
        cls: total_samples / (
            number_of_classes * count
        )
        for cls, count in class_counts.items()
    }

    sample_weights = (
        y_core
        .map(class_weights)
        .to_numpy()
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        subsample=0.80,
        random_state=42,
    )

    gb.fit(
        X_core,
        y_core,
        sample_weight=sample_weights,
    )

    gb_pred = gb.predict(
        X_val
    ).astype(int)

    gb_metrics = evaluate_predictions(
        y_val,
        gb_pred,
    )

    print("\nGRADIENT BOOSTING CLASSIFIER VALIDATION")
    print("-" * 70)

    for key, value in gb_metrics.items():
        print(
            f"{key:<20}: {value:.4f}"
        )

    # -----------------------------------------------------------------
    # Ordinal Gradient Boosting regression
    # -----------------------------------------------------------------
    print("\nTraining ordinal Gradient Boosting...")

    ordinal_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.80,
        loss="huber",
        random_state=42,
    )

    ordinal_model.fit(
        X_core,
        y_core.astype(float),
    )

    ordinal_continuous = (
        ordinal_model.predict(
            X_val
        )
    )

    ordinal_pred = np.clip(
        np.rint(
            ordinal_continuous
        ).astype(int),
        1,
        5,
    )

    ordinal_metrics = evaluate_predictions(
        y_val,
        ordinal_pred,
    )

    print("\nORDINAL REGRESSION VALIDATION")
    print("-" * 70)

    for key, value in ordinal_metrics.items():
        print(
            f"{key:<20}: {value:.4f}"
        )

    # -----------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------
    results = pd.DataFrame(
        [
            {
                "model": "random_forest_classifier",
                **rf_metrics,
            },
            {
                "model": "gradient_boosting_classifier",
                **gb_metrics,
            },
            {
                "model": "ordinal_gradient_boosting",
                **ordinal_metrics,
            },
        ]
    )

    # -----------------------------------------------------------------
    # Primary selection criterion:
    # Macro F1
    #
    # Macro F1 gives every review class equal importance.
    # -----------------------------------------------------------------
    best_index = results["macro_f1"].idxmax()

    results["selection_status"] = "Not Selected"

    results.loc[
        best_index,
        "selection_status",
    ] = "Selected"

    selected_model = results.loc[
        best_index,
        "model",
    ]

    # -----------------------------------------------------------------
    # Save validation results
    # -----------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------
    print("\nMODEL VALIDATION COMPARISON")
    print("-" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSELECTED MODEL")
    print("-" * 70)
    print(selected_model)

    print(
        f"Validation Macro F1: "
        f"{results.loc[best_index, 'macro_f1']:.4f}"
    )

    print(
        f"Validation Balanced Accuracy: "
        f"{results.loc[best_index, 'balanced_accuracy']:.4f}"
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
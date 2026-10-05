from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "ml" / "delivery_features_train.csv"
TEST_PATH = BASE_DIR / "data" / "ml" / "delivery_features_test.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "late_delivery_threshold_results.csv"
)


def prepare_data(df):
    excluded = [
        "order_id",
        "delivery_days",
        "late_delivery_flag",
    ]

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return df[features], df["late_delivery_flag"]


def evaluate_thresholds(y_true, probabilities):
    rows = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("LATE DELIVERY THRESHOLD TUNING")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # ---------------------------------------------------------------
    # Chronological validation split INSIDE training period.
    # ---------------------------------------------------------------
    split_index = int(len(train_df) * 0.80)

    train_core = train_df.iloc[:split_index].copy()
    validation = train_df.iloc[split_index:].copy()

    print("\nTIME-AWARE VALIDATION SPLIT")
    print("-" * 70)
    print(f"Training-core rows: {len(train_core):,}")
    print(f"Validation rows:    {len(validation):,}")

    print(
        f"Training-core period: "
        f"{train_core.purchase_year.min()} "
        f"to "
        f"{train_core.purchase_year.max()}"
    )

    print(
        f"Validation period starts at row: "
        f"{split_index:,}"
    )

    X_core, y_core = prepare_data(train_core)
    X_val, y_val = prepare_data(validation)

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------
    print("\nTraining Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_core, y_core)

    rf_val_prob = rf.predict_proba(X_val)[:, 1]

    rf_roc_auc = roc_auc_score(
        y_val,
        rf_val_prob,
    )

    rf_pr_auc = average_precision_score(
        y_val,
        rf_val_prob,
    )

    rf_thresholds = evaluate_thresholds(
        y_val,
        rf_val_prob,
    )

    rf_best = rf_thresholds.loc[
        rf_thresholds["f1"].idxmax()
    ].copy()

    print("\nRANDOM FOREST VALIDATION")
    print("-" * 70)
    print(f"ROC-AUC: {rf_roc_auc:.4f}")
    print(f"PR-AUC:  {rf_pr_auc:.4f}")
    print(f"Best threshold: {rf_best['threshold']:.2f}")
    print(f"Precision:      {rf_best['precision']:.4f}")
    print(f"Recall:         {rf_best['recall']:.4f}")
    print(f"F1:             {rf_best['f1']:.4f}")

    # ---------------------------------------------------------------
    # Gradient Boosting
    # ---------------------------------------------------------------
    print("\nTraining Gradient Boosting...")

    positive_count = (y_core == 1).sum()
    negative_count = (y_core == 0).sum()

    total_count = len(y_core)
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
        y_core == 1,
        weight_positive,
        weight_negative,
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

    gb_val_prob = gb.predict_proba(X_val)[:, 1]

    gb_roc_auc = roc_auc_score(
        y_val,
        gb_val_prob,
    )

    gb_pr_auc = average_precision_score(
        y_val,
        gb_val_prob,
    )

    gb_thresholds = evaluate_thresholds(
        y_val,
        gb_val_prob,
    )

    gb_best = gb_thresholds.loc[
        gb_thresholds["f1"].idxmax()
    ].copy()

    print("\nGRADIENT BOOSTING VALIDATION")
    print("-" * 70)
    print(f"ROC-AUC: {gb_roc_auc:.4f}")
    print(f"PR-AUC:  {gb_pr_auc:.4f}")
    print(f"Best threshold: {gb_best['threshold']:.2f}")
    print(f"Precision:      {gb_best['precision']:.4f}")
    print(f"Recall:         {gb_best['recall']:.4f}")
    print(f"F1:             {gb_best['f1']:.4f}")

    # ---------------------------------------------------------------
    # Compare models.
    # ---------------------------------------------------------------
    results = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "validation_roc_auc": rf_roc_auc,
                "validation_pr_auc": rf_pr_auc,
                "best_threshold": rf_best["threshold"],
                "precision": rf_best["precision"],
                "recall": rf_best["recall"],
                "f1": rf_best["f1"],
            },
            {
                "model": "gradient_boosting",
                "validation_roc_auc": gb_roc_auc,
                "validation_pr_auc": gb_pr_auc,
                "best_threshold": gb_best["threshold"],
                "precision": gb_best["precision"],
                "recall": gb_best["recall"],
                "f1": gb_best["f1"],
            },
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nMODEL COMPARISON")
    print("-" * 70)
    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
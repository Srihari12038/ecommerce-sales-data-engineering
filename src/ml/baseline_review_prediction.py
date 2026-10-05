from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "review_baseline_results.csv"
)


def main():
    print("=" * 70)
    print("REVIEW SCORE PREDICTION BASELINE")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "review_score"

    y_train = train_df[target]
    y_test = test_df[target]

    # -----------------------------------------------------------------
    # Majority-class baseline
    # -----------------------------------------------------------------
    majority_class = int(
        y_train.mode().iloc[0]
    )

    predictions = np.full(
        len(y_test),
        majority_class,
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

    print("\nBASELINE")
    print("-" * 70)
    print(
        f"Majority prediction: review score "
        f"{majority_class}"
    )

    print("\nPERFORMANCE")
    print("-" * 70)
    print(
        f"Accuracy:           {accuracy:.4f}"
    )
    print(
        f"Balanced Accuracy:  {balanced_accuracy:.4f}"
    )
    print(
        f"Macro F1:           {macro_f1:.4f}"
    )
    print(
        f"Weighted F1:        {weighted_f1:.4f}"
    )

    # -----------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------
    print("\nCONFUSION MATRIX")
    print("-" * 70)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3, 4, 5],
    )

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
    # Target distribution
    # -----------------------------------------------------------------
    print("TEST TARGET DISTRIBUTION")
    print("-" * 70)

    distribution = (
        y_test
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        distribution.round(2)
    )

    # -----------------------------------------------------------------
    # Save evaluation
    # -----------------------------------------------------------------
    results = pd.DataFrame(
        [
            {
                "problem": "review_score_classification",
                "model": "majority_baseline",
                "majority_class": majority_class,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
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

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
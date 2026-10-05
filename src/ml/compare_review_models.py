from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    BASE_DIR
    / "data"
    / "ml"
    / "evaluation"
)

OUTPUT_PATH = (
    EVALUATION_DIR
    / "review_model_comparison.csv"
)


def load_result(filename):
    path = EVALUATION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {path}"
        )

    return pd.read_csv(path)


def main():
    print("=" * 70)
    print("REVIEW MODEL COMPARISON")
    print("=" * 70)

    baseline = load_result(
        "review_baseline_results.csv"
    )

    random_forest = load_result(
        "review_random_forest_results.csv"
    )

    gradient_boosting = load_result(
        "review_gradient_boosting_results.csv"
    )

    ordinal = load_result(
        "review_ordinal_results.csv"
    )

    # ---------------------------------------------------------------
    # Normalize columns
    # ---------------------------------------------------------------
    baseline["model_family"] = "baseline"
    random_forest["model_family"] = "classification"
    gradient_boosting["model_family"] = "classification"
    ordinal["model_family"] = "ordinal_regression"

    comparison = pd.concat(
        [
            baseline,
            random_forest,
            gradient_boosting,
            ordinal,
        ],
        ignore_index=True,
        sort=False,
    )

    # ---------------------------------------------------------------
    # Determine best exact classifier
    # ---------------------------------------------------------------
    classifier_rows = comparison[
        comparison["model"].isin(
            [
                "random_forest",
                "gradient_boosting",
            ]
        )
    ].copy()

    best_classifier = classifier_rows.loc[
        classifier_rows["macro_f1"].idxmax()
    ]

    # ---------------------------------------------------------------
    # Determine best ordinal regression model by score MAE
    # ---------------------------------------------------------------
    ordinal_rows = comparison[
        comparison["model"].isin(
            [
                "gradient_boosting_regression",
            ]
        )
    ].copy()

    if not ordinal_rows.empty:
        best_ordinal = ordinal_rows.loc[
            ordinal_rows["score_mae"].idxmin()
        ]
    else:
        best_ordinal = None

    # ---------------------------------------------------------------
    # Add selection status
    # ---------------------------------------------------------------
    comparison["selection_status"] = "Not Selected"

    comparison.loc[
        comparison["model"]
        == best_classifier["model"],
        "selection_status",
    ] = "Best Exact Classifier"

    if best_ordinal is not None:
        comparison.loc[
            comparison["model"]
            == best_ordinal["model"],
            "selection_status",
        ] = "Best Ordinal Model"

    # The majority baseline is retained for reference.
    comparison.loc[
        comparison["model"]
        == "majority_baseline",
        "selection_status",
    ] = "Baseline Reference"

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------
    print("\nMODEL COMPARISON")
    print("-" * 70)

    columns_to_show = [
        "model",
        "model_family",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "score_mae",
        "quadratic_kappa",
        "selection_status",
    ]

    available_columns = [
        column
        for column in columns_to_show
        if column in comparison.columns
    ]

    print(
        comparison[available_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSELECTED EXACT CLASSIFIER")
    print("-" * 70)
    print(
        f"Model: {best_classifier['model']}"
    )
    print(
        f"Macro F1: "
        f"{best_classifier['macro_f1']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{best_classifier['balanced_accuracy']:.4f}"
    )

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 70)
    print(
        "No tested model demonstrates strong predictive "
        "performance for exact 1-5 review scores."
    )
    print(
        "The Random Forest is the best exact classifier "
        "by Macro F1, but performance remains weak."
    )
    print(
        "The ordinal regression model does not outperform "
        "the majority baseline on score error."
    )

    print("\nSAVED")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
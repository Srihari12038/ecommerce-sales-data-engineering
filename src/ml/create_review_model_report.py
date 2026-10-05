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
    / "review_model_report.csv"
)


def main():
    print("=" * 70)
    print("CREATING REVIEW MODEL REPORT")
    print("=" * 70)

    validation_path = (
        EVALUATION_DIR
        / "review_validation_results.csv"
    )

    final_path = (
        EVALUATION_DIR
        / "review_final_results.csv"
    )

    comparison_path = (
        EVALUATION_DIR
        / "review_model_comparison.csv"
    )

    validation = pd.read_csv(
        validation_path
    )

    final = pd.read_csv(
        final_path
    )

    comparison = pd.read_csv(
        comparison_path
    )

    # ---------------------------------------------------------------
    # Selected model
    # ---------------------------------------------------------------
    selected_validation = validation.loc[
        validation["selection_status"]
        == "Selected"
    ].iloc[0]

    selected_model = selected_validation["model"]

    # ---------------------------------------------------------------
    # Final test result
    # ---------------------------------------------------------------
    final_row = final.iloc[0]

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    report = pd.DataFrame(
        [
            {
                "module": "Review Satisfaction Prediction",
                "selected_model": selected_model,
                "selection_basis": (
                    "Chronological validation Macro F1"
                ),
                "validation_macro_f1": (
                    selected_validation["macro_f1"]
                ),
                "validation_balanced_accuracy": (
                    selected_validation[
                        "balanced_accuracy"
                    ]
                ),
                "validation_quadratic_kappa": (
                    selected_validation[
                        "quadratic_kappa"
                    ]
                ),
                "final_test_accuracy": (
                    final_row["accuracy"]
                ),
                "final_test_balanced_accuracy": (
                    final_row["balanced_accuracy"]
                ),
                "final_test_macro_f1": (
                    final_row["macro_f1"]
                ),
                "final_test_weighted_f1": (
                    final_row["weighted_f1"]
                ),
                "final_test_score_mae": (
                    final_row["score_mae"]
                ),
                "final_test_quadratic_kappa": (
                    final_row["quadratic_kappa"]
                ),
                "baseline_macro_f1": (
                    final_row[
                        "baseline_macro_f1"
                    ]
                ),
                "baseline_balanced_accuracy": (
                    final_row[
                        "baseline_balanced_accuracy"
                    ]
                ),
                "model_status": (
                    "Selected but limited predictive power"
                ),
                "business_interpretation": (
                    "Order-time transactional features "
                    "provide limited ability to predict "
                    "exact customer review scores. "
                    "The model improves over the majority "
                    "baseline on Macro F1 and balanced "
                    "accuracy, but performance remains weak."
                ),
            }
        ]
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSELECTED MODEL")
    print("-" * 70)
    print(selected_model)

    print("\nFINAL TEST RESULTS")
    print("-" * 70)

    print(
        f"Accuracy:            "
        f"{final_row['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:   "
        f"{final_row['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:            "
        f"{final_row['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:         "
        f"{final_row['weighted_f1']:.4f}"
    )

    print(
        f"Score MAE:           "
        f"{final_row['score_mae']:.4f}"
    )

    print(
        f"Quadratic Kappa:     "
        f"{final_row['quadratic_kappa']:.4f}"
    )

    print("\nMODEL CONCLUSION")
    print("-" * 70)
    print(
        "Selected for the project, but not considered "
        "a high-performance predictive model."
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

WALK_FORWARD_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "walk_forward_results.csv"
)

MODEL_COMPARISON_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "model_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "forecasting_model_report.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Load walk-forward results
    # ---------------------------------------------------------

    walk_forward = pd.read_csv(
        WALK_FORWARD_FILE
    )

    # ---------------------------------------------------------
    # 2. Aggregate walk-forward metrics
    # ---------------------------------------------------------

    summary = (
        walk_forward
        .groupby("model")
        .agg(
            average_mae=("mae", "mean"),
            average_rmse=("rmse", "mean"),
            average_smape=("smape", "mean")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3. Determine best model
    # ---------------------------------------------------------

    best_model = (
        summary
        .loc[
            summary["average_mae"].idxmin(),
            "model"
        ]
    )

    summary["selection_status"] = (
        summary["model"]
        .apply(
            lambda model:
            "Selected"
            if model == best_model
            else "Not Selected"
        )
    )

    # ---------------------------------------------------------
    # 4. Add validation methodology
    # ---------------------------------------------------------

    summary["validation_method"] = (
        "Walk-Forward Validation"
    )

    summary["target"] = (
        "Monthly Total Sales"
    )

    # ---------------------------------------------------------
    # 5. Reorder columns
    # ---------------------------------------------------------

    report = summary[
        [
            "model",
            "validation_method",
            "target",
            "average_mae",
            "average_rmse",
            "average_smape",
            "selection_status"
        ]
    ].copy()

    # ---------------------------------------------------------
    # 6. Save report
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("FORECASTING MODEL EVALUATION REPORT")
    print("=" * 80)

    print("\nModel Comparison")
    print("-" * 80)

    print(
        report.to_string(
            index=False
        )
    )

    print("\nSelected Model")
    print("-" * 80)

    print(best_model)

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
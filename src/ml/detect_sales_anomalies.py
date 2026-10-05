import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "monthly_sales_ml.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "ml"
    / "evaluation"
    / "sales_anomalies.csv"
)


def main():

    # ---------------------------------------------------------
    # 1. Load monthly sales
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2. Basic statistics
    # ---------------------------------------------------------

    mean_sales = df["total_sales"].mean()

    median_sales = df["total_sales"].median()

    std_sales = df["total_sales"].std()

    # ---------------------------------------------------------
    # 3. Z-score detection
    # ---------------------------------------------------------

    df["z_score"] = (
        (
            df["total_sales"]
            - mean_sales
        )
        / std_sales
    )

    df["anomaly_zscore"] = (
        df["z_score"].abs() > 2
    )

    # ---------------------------------------------------------
    # 4. IQR detection
    # ---------------------------------------------------------

    q1 = df["total_sales"].quantile(0.25)

    q3 = df["total_sales"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    df["anomaly_iqr"] = (
        (df["total_sales"] < lower_bound)
        |
        (df["total_sales"] > upper_bound)
    )

    # ---------------------------------------------------------
    # 5. Month-over-month change
    # ---------------------------------------------------------

    df["sales_change_pct"] = (
        df["total_sales"]
        .pct_change()
        * 100
    )

    # ---------------------------------------------------------
    # 6. Business-rule anomaly
    # ---------------------------------------------------------

    # Flag an extreme monthly movement greater than 80%.
    #
    # This is a business monitoring rule rather than a
    # statistical claim.

    df["anomaly_mom"] = (
        df["sales_change_pct"].abs() > 80
    )

    # ---------------------------------------------------------
    # 7. Combined anomaly flag
    # ---------------------------------------------------------

    df["is_anomaly"] = (
        df["anomaly_zscore"]
        |
        df["anomaly_iqr"]
        |
        df["anomaly_mom"]
    )

    # ---------------------------------------------------------
    # 8. Classify anomaly
    # ---------------------------------------------------------

    df["anomaly_type"] = "Normal"

    df.loc[
        df["sales_change_pct"] <= -80,
        "anomaly_type"
    ] = "Extreme Sales Drop"

    df.loc[
        df["sales_change_pct"] >= 80,
        "anomaly_type"
    ] = "Extreme Sales Increase"

    # Statistical anomalies where there is no
    # extreme MoM movement.

    statistical_only = (
        (df["anomaly_zscore"] | df["anomaly_iqr"])
        &
        (~df["anomaly_mom"])
    )

    df.loc[
        statistical_only,
        "anomaly_type"
    ] = "Statistical Anomaly"

    # ---------------------------------------------------------
    # 9. Save results
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_columns = [
        "date",
        "total_sales",
        "sales_change_pct",
        "z_score",
        "anomaly_zscore",
        "anomaly_iqr",
        "anomaly_mom",
        "is_anomaly",
        "anomaly_type"
    ]

    df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 10. Display
    # ---------------------------------------------------------

    print("=" * 80)
    print("SALES ANOMALY DETECTION")
    print("=" * 80)

    print("\nStatistics")
    print("-" * 80)

    print(
        f"Mean sales   : ₹{mean_sales:,.2f}"
    )

    print(
        f"Median sales : ₹{median_sales:,.2f}"
    )

    print(
        f"Std deviation: ₹{std_sales:,.2f}"
    )

    print("\nIQR Boundaries")
    print("-" * 80)

    print(
        f"Lower bound: ₹{lower_bound:,.2f}"
    )

    print(
        f"Upper bound: ₹{upper_bound:,.2f}"
    )

    print("\nBusiness Rule")
    print("-" * 80)

    print(
        "Extreme month-over-month change threshold: 80%"
    )

    print("\nDetected Anomalies")
    print("-" * 80)

    anomalies = df[
        df["is_anomaly"]
    ]

    if anomalies.empty:

        print("No anomalies detected.")

    else:

        print(
            anomalies[
                [
                    "date",
                    "total_sales",
                    "sales_change_pct",
                    "anomaly_type"
                ]
            ].to_string(
                index=False
            )
        )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
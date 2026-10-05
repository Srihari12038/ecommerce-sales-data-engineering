from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = BASE_DIR / "data" / "ml" / "evaluation"
ML_DIR = BASE_DIR / "data" / "ml"
OUTPUT_DIR = ML_DIR / "insights"

OUTPUT_FILE = OUTPUT_DIR / "ai_business_insights.csv"


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path: Path):
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            print(f"[WARNING] Empty file: {path}")
            return None

        return df

    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return None


def add_insight(
    rows,
    module,
    insight_type,
    metric,
    value,
    finding,
    business_impact,
    recommended_action,
    priority,
):
    rows.append(
        {
            "module": module,
            "insight_type": insight_type,
            "metric": metric,
            "value": value,
            "finding": finding,
            "business_impact": business_impact,
            "recommended_action": recommended_action,
            "priority": priority,
        }
    )


# ============================================================
# LOAD SOURCE FILES
# ============================================================

forecasting_report = safe_read_csv(
    EVALUATION_DIR / "forecasting_model_report.csv"
)

walk_forward = safe_read_csv(
    EVALUATION_DIR / "walk_forward_results.csv"
)

customer_segments = safe_read_csv(
    ML_DIR / "customer_segment_analysis.csv"
)

delivery_report = safe_read_csv(
    EVALUATION_DIR / "delivery_model_report.csv"
)

review_report = safe_read_csv(
    EVALUATION_DIR / "review_model_report.csv"
)

anomaly_report = safe_read_csv(
    EVALUATION_DIR / "order_anomaly_report.csv"
)

recommendation_report = safe_read_csv(
    EVALUATION_DIR / "product_recommender_report.csv"
)


# ============================================================
# INSIGHT COLLECTION
# ============================================================

insights = []


# ============================================================
# 1. SALES FORECASTING
# ============================================================

if forecasting_report is not None:

    selected_rows = forecasting_report[
        forecasting_report["selection_status"]
        .astype(str)
        .str.lower()
        == "selected"
    ]

    if not selected_rows.empty:

        row = selected_rows.iloc[0]

        model_name = str(row["model"])
        avg_mae = float(row["average_mae"])
        avg_rmse = float(row["average_rmse"])
        avg_smape = float(row["average_smape"])

        add_insight(
            insights,
            module="Sales Forecasting",
            insight_type="Forecasting Performance",
            metric="Average MAE",
            value=round(avg_mae, 2),
            finding=(
                f"The selected forecasting model is {model_name} "
                f"with an average walk-forward MAE of "
                f"{avg_mae:,.2f}."
            ),
            business_impact=(
                "Forecast accuracy determines how reliably future "
                "sales can support planning and inventory decisions."
            ),
            recommended_action=(
                "Use the selected forecasting model as a planning "
                "baseline and monitor forecast error over time."
            ),
            priority="High",
        )

        add_insight(
            insights,
            module="Sales Forecasting",
            insight_type="Forecasting Error",
            metric="Average sMAPE",
            value=round(avg_smape, 4),
            finding=(
                f"The selected forecasting model has an average "
                f"walk-forward sMAPE of {avg_smape:.2f}%."
            ),
            business_impact=(
                "Forecast error should be considered when using "
                "predictions for operational planning."
            ),
            recommended_action=(
                "Combine forecasts with historical sales trends "
                "and business context before major decisions."
            ),
            priority="Medium",
        )

    else:
        # Fallback: choose the lowest MAE model reported.
        valid_report = forecasting_report.dropna(
            subset=["average_mae"]
        ).copy()

        if not valid_report.empty:

            row = valid_report.sort_values(
                "average_mae"
            ).iloc[0]

            model_name = str(row["model"])
            avg_mae = float(row["average_mae"])
            avg_smape = float(row["average_smape"])

            add_insight(
                insights,
                module="Sales Forecasting",
                insight_type="Forecasting Performance",
                metric="Best Reported Average MAE",
                value=round(avg_mae, 2),
                finding=(
                    f"{model_name} has the lowest reported average "
                    f"walk-forward MAE of {avg_mae:,.2f} in the "
                    f"available forecasting report."
                ),
                business_impact=(
                    "Forecast error directly affects the reliability "
                    "of sales planning."
                ),
                recommended_action=(
                    "Use the validated forecasting comparison when "
                    "selecting a production planning baseline."
                ),
                priority="High",
            )

            add_insight(
                insights,
                module="Sales Forecasting",
                insight_type="Forecasting Error",
                metric="Average sMAPE",
                value=round(avg_smape, 4),
                finding=(
                    f"The best reported forecasting configuration "
                    f"has an average sMAPE of {avg_smape:.2f}%."
                ),
                business_impact=(
                    "Forecast uncertainty should be considered when "
                    "using projected sales for planning."
                ),
                recommended_action=(
                    "Use forecasts together with historical trends "
                    "rather than as a standalone decision signal."
                ),
                priority="Medium",
            )


# ============================================================
# 2. CUSTOMER SEGMENTATION
# ============================================================

if customer_segments is not None:

    high_value = customer_segments[
        customer_segments["segment"].astype(str).str.lower()
        == "high value customers"
    ]

    standard = customer_segments[
        customer_segments["segment"].astype(str).str.lower()
        == "standard customers"
    ]

    if not high_value.empty:

        row = high_value.iloc[0]

        customers = int(row["customers"])
        customer_percentage = float(row["customer_percentage"])
        sales_percentage = float(row["sales_percentage"])
        monetary = float(row["average_monetary_value"])
        total_sales = float(row["total_sales"])

        add_insight(
            insights,
            module="Customer Segmentation",
            insight_type="High Value Segment",
            metric="High Value Customer Sales Share",
            value=round(sales_percentage, 2),
            finding=(
                f"High Value Customers represent "
                f"{customer_percentage:.2f}% of customers "
                f"but contribute {sales_percentage:.2f}% of sales."
            ),
            business_impact=(
                "A relatively small customer segment contributes a "
                "disproportionate share of sales."
            ),
            recommended_action=(
                "Prioritize retention, premium offers, cross-selling "
                "and personalized campaigns for this segment."
            ),
            priority="High",
        )

        add_insight(
            insights,
            module="Customer Segmentation",
            insight_type="Customer Value",
            metric="Average Monetary Value",
            value=round(monetary, 2),
            finding=(
                f"High Value Customers have an average monetary "
                f"value of {monetary:.2f}."
            ),
            business_impact=(
                "Higher-value customers provide an important "
                "opportunity for retention and cross-selling."
            ),
            recommended_action=(
                "Design targeted campaigns around premium products "
                "and repeat-purchase opportunities."
            ),
            priority="Medium",
        )

    if not standard.empty:

        row = standard.iloc[0]

        standard_percentage = float(row["customer_percentage"])
        standard_sales_percentage = float(row["sales_percentage"])
        standard_monetary = float(
            row["average_monetary_value"]
        )

        add_insight(
            insights,
            module="Customer Segmentation",
            insight_type="Growth Opportunity",
            metric="Standard Customer Sales Share",
            value=round(standard_sales_percentage, 2),
            finding=(
                f"Standard Customers represent "
                f"{standard_percentage:.2f}% of customers "
                f"and contribute {standard_sales_percentage:.2f}% "
                f"of sales."
            ),
            business_impact=(
                "The large Standard segment represents the broadest "
                "opportunity for increasing customer value."
            ),
            recommended_action=(
                "Use targeted promotions, product recommendations "
                "and incentives to increase order value."
            ),
            priority="High",
        )


# ============================================================
# 3. DELIVERY PREDICTION
# ============================================================

if delivery_report is not None:

    row = delivery_report.iloc[0]

    # Identify MAE column.
    mae_column = None

    for column in delivery_report.columns:
        if str(column).lower() in {
            "mae",
            "test_mae",
            "final_test_mae",
            "selected_mae",
        }:
            mae_column = column
            break

    if mae_column is not None:

        mae = float(row[mae_column])

        add_insight(
            insights,
            module="Delivery Prediction",
            insight_type="Operational Risk",
            metric="Delivery MAE",
            value=round(mae, 4),
            finding=(
                f"The selected delivery model has an average "
                f"absolute prediction error of {mae:.2f} days."
            ),
            business_impact=(
                "Delivery prediction can support logistics planning "
                "and customer expectation management."
            ),
            recommended_action=(
                "Use predicted delivery duration and late-risk "
                "information to prioritize potentially delayed orders."
            ),
            priority="High",
        )


# ============================================================
# 4. REVIEW SATISFACTION PREDICTION
# ============================================================

if review_report is not None:

    row = review_report.iloc[0]

    final_mae = float(
        row["final_test_score_mae"]
    )

    final_macro_f1 = float(
        row["final_test_macro_f1"]
    )

    validation_macro_f1 = float(
        row["validation_macro_f1"]
    )

    model_status = str(
        row["model_status"]
    )

    add_insight(
        insights,
        module="Review Satisfaction Prediction",
        insight_type="Prediction Performance",
        metric="Final Test Macro F1",
        value=round(final_macro_f1, 6),
        finding=(
            f"The selected Random Forest classifier achieved "
            f"a final test Macro F1 of {final_macro_f1:.4f}."
        ),
        business_impact=(
            "The model can provide an exploratory satisfaction "
            "signal, but class-level prediction remains limited."
        ),
        recommended_action=(
            "Use predicted satisfaction as a supporting analytical "
            "signal and combine it with actual customer reviews."
        ),
        priority="Medium",
    )

    add_insight(
        insights,
        module="Review Satisfaction Prediction",
        insight_type="Model Limitation",
        metric="Final Test Score MAE",
        value=round(final_mae, 4),
        finding=(
            f"The model has a final test review-score MAE of "
            f"{final_mae:.3f}. The model status is: {model_status}."
        ),
        business_impact=(
            "Review-score prediction should not be treated as a "
            "high-confidence individual customer prediction."
        ),
        recommended_action=(
            "Use the model primarily for exploratory analysis and "
            "identify areas where additional behavioral features "
            "could improve prediction."
        ),
        priority="Medium",
    )


# ============================================================
# 5. ORDER ANOMALY DETECTION
# ============================================================

if anomaly_report is not None:

    row = anomaly_report.iloc[0]

    anomaly_count = None
    anomaly_rate = None

    for column in anomaly_report.columns:

        column_lower = str(column).lower()

        if column_lower in {
            "anomalous_orders",
            "anomaly_count",
            "num_anomalies",
            "anomalies",
        }:
            anomaly_count = int(row[column])

        if column_lower in {
            "anomaly_rate",
            "anomalous_percentage",
            "anomaly_percentage",
        }:
            anomaly_rate = float(row[column])

    if anomaly_count is not None:

        add_insight(
            insights,
            module="Order Anomaly Detection",
            insight_type="Transactional Anomaly",
            metric="Anomalous Orders",
            value=anomaly_count,
            finding=(
                f"{anomaly_count:,} orders were identified as "
                f"transactional anomalies."
            ),
            business_impact=(
                "Unusual order structures and unusually large "
                "transactions may require operational review."
            ),
            recommended_action=(
                "Prioritize anomalous orders for manual validation "
                "and operational investigation."
            ),
            priority="High",
        )

    if anomaly_rate is not None:

        # Handle either decimal or percentage storage.
        displayed_rate = (
            anomaly_rate * 100
            if anomaly_rate <= 1
            else anomaly_rate
        )

        add_insight(
            insights,
            module="Order Anomaly Detection",
            insight_type="Anomaly Rate",
            metric="Anomaly Rate",
            value=round(displayed_rate, 4),
            finding=(
                f"The detected transactional anomaly rate is "
                f"{displayed_rate:.2f}%."
            ),
            business_impact=(
                "A small set of unusual transactions can be "
                "isolated for investigation."
            ),
            recommended_action=(
                "Use anomaly flags as a review queue rather than "
                "automatically classifying transactions as fraud."
            ),
            priority="High",
        )


# ============================================================
# 6. PRODUCT RECOMMENDATION
# ============================================================

if recommendation_report is not None:

    row = recommendation_report.iloc[0]

    precision = float(row["precision_at_k"])
    recall = float(row["recall_at_k"])
    hit_rate = float(row["hit_rate_at_k"])
    coverage = float(row["source_coverage"])

    k = int(row["k"])

    add_insight(
        insights,
        module="Product Recommendation",
        insight_type="Recommendation Quality",
        metric=f"Precision@{k}",
        value=round(precision, 6),
        finding=(
            f"The item-to-item recommendation prototype achieved "
            f"Precision@{k} of {precision:.4f}."
        ),
        business_impact=(
            "The current co-purchase signal is sparse and is not "
            "sufficient for high-confidence automated recommendations."
        ),
        recommended_action=(
            "Retain the model as a baseline association-based "
            "recommendation prototype."
        ),
        priority="Medium",
    )

    add_insight(
        insights,
        module="Product Recommendation",
        insight_type="Recommendation Coverage",
        metric=f"Recall@{k}",
        value=round(recall, 6),
        finding=(
            f"The recommendation prototype achieved Recall@{k} "
            f"of {recall:.4f}."
        ),
        business_impact=(
            "Only a limited portion of relevant co-purchased "
            "products are recovered."
        ),
        recommended_action=(
            "Improve future versions with richer product and "
            "customer interaction signals."
        ),
        priority="Medium",
    )

    add_insight(
        insights,
        module="Product Recommendation",
        insight_type="Recommendation Reach",
        metric="Source Coverage",
        value=round(coverage, 6),
        finding=(
            f"The recommender provides association-based results "
            f"for approximately {coverage * 100:.2f}% of supported "
            f"source products."
        ),
        business_impact=(
            "Many products do not have sufficiently strong "
            "historical co-purchase relationships."
        ),
        recommended_action=(
            "Introduce fallback recommendations such as popular "
            "products for products without association rules."
        ),
        priority="Medium",
    )


# ============================================================
# 7. PLATFORM-LEVEL INSIGHT
# ============================================================

modules_detected = {
    insight["module"]
    for insight in insights
}

required_modules = {
    "Sales Forecasting",
    "Customer Segmentation",
    "Delivery Prediction",
    "Review Satisfaction Prediction",
    "Order Anomaly Detection",
    "Product Recommendation",
}

covered_modules = modules_detected.intersection(required_modules)

add_insight(
    insights,
    module="Platform",
    insight_type="Integrated Analytics",
    metric="ML Module Coverage",
    value=len(covered_modules),
    finding=(
        f"The AI Business Insights layer currently integrates "
        f"{len(covered_modules)} of {len(required_modules)} "
        f"completed ML modules."
    ),
    business_impact=(
        "Combining forecasting, segmentation, operational prediction, "
        "satisfaction analysis, anomaly detection and recommendation "
        "creates a unified decision-support layer."
    ),
    recommended_action=(
        "Use the integrated insight table as the analytical source "
        "for the final Power BI AI and ML reporting layer."
    ),
    priority="High",
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

insights_df = pd.DataFrame(insights)

if insights_df.empty:
    raise RuntimeError(
        "No business insights were generated."
    )


# ============================================================
# ADD ID
# ============================================================

insights_df.insert(
    0,
    "insight_id",
    range(1, len(insights_df) + 1)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

insights_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("AI BUSINESS INSIGHTS GENERATOR")
print("=" * 70)

print("\nINSIGHTS GENERATED")
print("-" * 70)
print(f"Total insights: {len(insights_df)}")

print("\nBY MODULE")
print("-" * 70)
print(
    insights_df["module"]
    .value_counts()
    .to_string()
)

print("\nPRIORITY")
print("-" * 70)
print(
    insights_df["priority"]
    .value_counts()
    .to_string()
)

print("\nML MODULE COVERAGE")
print("-" * 70)

for module in sorted(required_modules):
    status = (
        "Included"
        if module in covered_modules
        else "Missing"
    )

    print(
        f"{module:<35} {status}"
    )

print("\nSAVED")
print("-" * 70)
print(OUTPUT_FILE)

print("=" * 70)
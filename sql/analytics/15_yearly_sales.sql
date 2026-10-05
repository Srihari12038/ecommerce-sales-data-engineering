-- ============================================================
-- 15 YEARLY SALES ANALYSIS
-- ============================================================

WITH yearly_sales AS (
    SELECT
        d.year,

        ROUND(SUM(f.price), 2) AS total_sales,

        COUNT(DISTINCT f.order_id) AS total_orders,

        COUNT(*) AS total_items

    FROM fact_sales f

    JOIN dim_date d
        ON f.order_date_key = d.date_key

    GROUP BY
        d.year
),

yearly_growth AS (
    SELECT
        year,
        total_sales,
        total_orders,
        total_items,

        LAG(total_sales) OVER (
            ORDER BY year
        ) AS previous_year_sales

    FROM yearly_sales
)

SELECT
    year,

    total_sales,

    total_orders,

    total_items,

    previous_year_sales,

    ROUND(
        (
            (total_sales - previous_year_sales)
            * 100.0
            / previous_year_sales
        ),
        2
    ) AS sales_growth_percentage

FROM yearly_growth

ORDER BY
    year;
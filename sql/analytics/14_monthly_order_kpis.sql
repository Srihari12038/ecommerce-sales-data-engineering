-- ============================================================
-- 14 MONTHLY ORDER KPIs
-- ============================================================

SELECT
    d.year,
    d.month,
    d.month_name,

    COUNT(DISTINCT f.order_id) AS total_orders,

    COUNT(*) AS total_items,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(
        SUM(f.price) / COUNT(DISTINCT f.order_id),
        2
    ) AS average_order_value

FROM fact_sales f

JOIN dim_date d
    ON f.order_date_key = d.date_key

GROUP BY
    d.year,
    d.month,
    d.month_name

ORDER BY
    d.year,
    d.month;
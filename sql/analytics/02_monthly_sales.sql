-- ============================================================
-- 02 MONTHLY SALES TREND
-- ============================================================

SELECT
    d.year,
    d.month,
    d.month_name,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    ROUND(SUM(f.total_item_value), 2) AS total_item_value,

    COUNT(*) AS total_items,

    COUNT(DISTINCT f.order_id) AS total_orders

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
-- ============================================================
-- 20 EXECUTIVE KPI SUMMARY
-- ============================================================

SELECT

    ROUND(
        SUM(price),
        2
    ) AS total_sales,

    ROUND(
        SUM(freight_value),
        2
    ) AS total_freight,

    COUNT(*) AS total_items,

    COUNT(DISTINCT order_id) AS total_orders,

    COUNT(DISTINCT customer_key) AS active_customers,

    ROUND(
        SUM(price) /
        COUNT(DISTINCT order_id),
        2
    ) AS average_order_value

FROM fact_sales;
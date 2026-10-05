-- ============================================================
-- 05 TOP 10 CUSTOMERS BY SALES
-- ============================================================

SELECT
    c.customer_unique_id,

    c.customer_city,

    c.customer_state,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    COUNT(DISTINCT f.order_id) AS total_orders,

    COUNT(*) AS total_items,

    ROUND(
        SUM(f.price) / COUNT(DISTINCT f.order_id),
        2
    ) AS average_order_value

FROM fact_sales f

JOIN dim_customer c
    ON f.customer_key = c.customer_key

GROUP BY
    c.customer_unique_id,
    c.customer_city,
    c.customer_state

ORDER BY
    total_sales DESC

LIMIT 10;
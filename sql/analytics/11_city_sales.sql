-- ============================================================
-- 11 TOP 10 CUSTOMER CITIES BY SALES
-- ============================================================

SELECT
    c.customer_city AS city,
    c.customer_state AS state,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    COUNT(DISTINCT f.order_id) AS total_orders,

    COUNT(*) AS total_items

FROM fact_sales f

JOIN dim_customer c
    ON f.customer_key = c.customer_key

GROUP BY
    c.customer_city,
    c.customer_state

ORDER BY
    total_sales DESC

LIMIT 10;
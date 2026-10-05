-- ============================================================
-- 12 TOP 10 SELLERS BY SALES
-- ============================================================

SELECT
    s.seller_id,

    s.seller_city,

    s.seller_state,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    COUNT(*) AS total_items,

    COUNT(DISTINCT f.order_id) AS total_orders,

    ROUND(
        SUM(f.price) / COUNT(DISTINCT f.order_id),
        2
    ) AS average_order_value

FROM fact_sales f

JOIN dim_seller s
    ON f.seller_key = s.seller_key

GROUP BY
    s.seller_id,
    s.seller_city,
    s.seller_state

ORDER BY
    total_sales DESC

LIMIT 10;
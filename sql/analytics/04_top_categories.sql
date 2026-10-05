-- ============================================================
-- 04 TOP 10 PRODUCT CATEGORIES BY SALES
-- ============================================================

SELECT
    COALESCE(
        p.product_category_name_english,
        'Unknown'
    ) AS product_category,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    COUNT(*) AS total_items,

    COUNT(DISTINCT f.order_id) AS total_orders,

    ROUND(
        SUM(f.price) / COUNT(DISTINCT f.order_id),
        2
    ) AS average_order_value

FROM fact_sales f

JOIN dim_product p
    ON f.product_key = p.product_key

GROUP BY
    p.product_category_name_english

ORDER BY
    total_sales DESC

LIMIT 10;
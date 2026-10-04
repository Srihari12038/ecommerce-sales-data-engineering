-- ============================================================
-- 03 TOP 10 PRODUCTS BY SALES
-- ============================================================

SELECT
    p.product_id,
    p.product_category_name_english AS product_category,

    ROUND(SUM(f.price), 2) AS total_sales,

    ROUND(SUM(f.freight_value), 2) AS total_freight,

    COUNT(*) AS total_items,

    COUNT(DISTINCT f.order_id) AS total_orders

FROM fact_sales f

JOIN dim_product p
    ON f.product_key = p.product_key

GROUP BY
    p.product_id,
    p.product_category_name_english

ORDER BY
    total_sales DESC

LIMIT 10;
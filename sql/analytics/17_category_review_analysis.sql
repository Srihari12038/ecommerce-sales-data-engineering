-- ============================================================
-- 17 PRODUCT CATEGORY + REVIEW ANALYSIS
-- ============================================================

SELECT
    COALESCE(
        p.product_category_name_english,
        'Unknown'
    ) AS product_category,

    COUNT(r.review_id) AS total_reviews,

    ROUND(
        AVG(r.review_score),
        2
    ) AS average_review_score,

    ROUND(
        SUM(f.price),
        2
    ) AS total_sales

FROM fact_sales f

JOIN dim_product p
    ON f.product_key = p.product_key

LEFT JOIN fact_reviews r
    ON f.order_id = r.order_id

GROUP BY
    p.product_category_name_english

HAVING
    COUNT(r.review_id) > 0

ORDER BY
    total_sales DESC

LIMIT 10;
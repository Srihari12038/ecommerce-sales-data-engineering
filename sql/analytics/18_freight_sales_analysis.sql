-- ============================================================
-- 18 FREIGHT VS SALES ANALYSIS
-- ============================================================

SELECT
    ROUND(SUM(price), 2) AS total_sales,

    ROUND(SUM(freight_value), 2) AS total_freight,

    ROUND(
        SUM(freight_value) * 100.0 /
        SUM(price),
        2
    ) AS freight_percentage_of_sales,

    ROUND(
        AVG(freight_value),
        2
    ) AS average_freight_per_item,

    COUNT(*) AS total_items

FROM fact_sales;
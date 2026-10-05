-- ============================================================
-- 16 REPEAT CUSTOMER ANALYSIS
-- ============================================================

WITH customer_orders AS (
    SELECT
        c.customer_unique_id,

        COUNT(DISTINCT f.order_id) AS total_orders,

        ROUND(
            SUM(f.price),
            2
        ) AS total_sales

    FROM fact_sales f

    JOIN dim_customer c
        ON f.customer_key = c.customer_key

    GROUP BY
        c.customer_unique_id
)

SELECT
    CASE
        WHEN total_orders = 1
            THEN 'One-time Customer'
        ELSE 'Repeat Customer'
    END AS customer_type,

    COUNT(*) AS total_customers,

    ROUND(
        SUM(total_sales),
        2
    ) AS total_sales,

    ROUND(
        AVG(total_sales),
        2
    ) AS average_customer_sales

FROM customer_orders

GROUP BY
    customer_type

ORDER BY
    total_sales DESC;
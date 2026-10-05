-- ============================================================
-- 13 ORDER STATUS ANALYSIS
-- ============================================================

SELECT
    order_status,

    COUNT(*) AS total_orders,

    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM dim_order),
        2
    ) AS percentage_of_orders

FROM dim_order

GROUP BY
    order_status

ORDER BY
    total_orders DESC;
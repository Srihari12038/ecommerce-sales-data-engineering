-- ============================================================
-- 08 ORDER DELIVERY PERFORMANCE
-- ============================================================

SELECT
    ROUND(AVG(delivery_days), 2) AS average_delivery_days,

    MIN(delivery_days) AS minimum_delivery_days,

    MAX(delivery_days) AS maximum_delivery_days,

    COUNT(*) AS delivered_orders

FROM dim_order

WHERE delivery_days IS NOT NULL;
-- ============================================================
-- 19 PAYMENT + ORDER ANALYSIS
-- ============================================================

SELECT
    p.payment_type,

    COUNT(DISTINCT p.order_id) AS total_orders,

    COUNT(*) AS total_payment_records,

    ROUND(
        SUM(p.payment_value),
        2
    ) AS total_payment_value,

    ROUND(
        AVG(p.payment_value),
        2
    ) AS average_payment_value,

    ROUND(
        AVG(p.payment_installments),
        2
    ) AS average_installments

FROM fact_payments p

GROUP BY
    p.payment_type

ORDER BY
    total_payment_value DESC;
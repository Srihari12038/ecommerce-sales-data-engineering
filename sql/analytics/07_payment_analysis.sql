-- ============================================================
-- 07 PAYMENT METHOD ANALYSIS
-- ============================================================

SELECT
    payment_type,

    COUNT(*) AS total_transactions,

    ROUND(SUM(payment_value), 2) AS total_payment_value,

    ROUND(AVG(payment_value), 2) AS average_payment_value,

    ROUND(AVG(payment_installments), 2) AS average_installments,

    MAX(payment_installments) AS max_installments

FROM fact_payments

GROUP BY
    payment_type

ORDER BY
    total_payment_value DESC;
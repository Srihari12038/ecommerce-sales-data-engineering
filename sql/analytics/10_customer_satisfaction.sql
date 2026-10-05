-- ============================================================
-- 10 OVERALL CUSTOMER SATISFACTION KPI
-- ============================================================

SELECT
    ROUND(AVG(review_score), 2) AS average_review_score,

    COUNT(*) AS total_reviews,

    SUM(
        CASE
            WHEN review_score >= 4 THEN 1
            ELSE 0
        END
    ) AS positive_reviews,

    ROUND(
        SUM(
            CASE
                WHEN review_score >= 4 THEN 1
                ELSE 0
            END
        ) * 100.0 / COUNT(*),
        2
    ) AS positive_review_percentage

FROM fact_reviews;
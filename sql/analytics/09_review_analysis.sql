-- ============================================================
-- 09 REVIEW / CUSTOMER SATISFACTION ANALYSIS
-- ============================================================

SELECT
    review_score,

    COUNT(*) AS total_reviews,

    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM fact_reviews),
        2
    ) AS percentage_of_reviews

FROM fact_reviews

GROUP BY
    review_score

ORDER BY
    review_score;
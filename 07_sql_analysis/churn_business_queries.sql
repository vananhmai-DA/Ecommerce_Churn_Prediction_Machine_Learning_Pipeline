/*
===============================================================================
Churn Business SQL Analysis
Project: Ecommerce Churn Prediction Pipeline

Purpose:
This file contains business-oriented SQL queries used to analyze churn behavior,
customer risk segments, and key retention opportunities from the Gold layer.

Main source table:
analytics_gold.gold_churn_model_input

Note:
Prediction outputs are generated in Python and saved as CSV files. If prediction
results are later loaded back into PostgreSQL, additional SQL analysis can be
performed on churn risk levels directly in the database.
===============================================================================
*/


/*
-------------------------------------------------------------------------------
1. Overall churn rate
-------------------------------------------------------------------------------
Business question:
What percentage of customers churned?
*/

SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM analytics_gold.gold_churn_model_input;


/*
-------------------------------------------------------------------------------
2. Churn rate by complaint status
-------------------------------------------------------------------------------
Business question:
Do customers with complaints churn more often?
*/

SELECT
    complain,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM analytics_gold.gold_churn_model_input
GROUP BY complain
ORDER BY complain;


/*
-------------------------------------------------------------------------------
3. Churn rate by customer tenure group
-------------------------------------------------------------------------------
Business question:
Are newer customers more likely to churn?
*/

WITH tenure_grouped AS (
    SELECT
        customer_id,
        churn,
        CASE
            WHEN tenure <= 1 THEN '01. Very new: tenure <= 1'
            WHEN tenure <= 3 THEN '02. New: tenure 2-3'
            WHEN tenure <= 6 THEN '03. Early: tenure 4-6'
            WHEN tenure <= 12 THEN '04. Established: tenure 7-12'
            ELSE '05. Loyal: tenure > 12'
        END AS tenure_group
    FROM analytics_gold.gold_churn_model_input
)

SELECT
    tenure_group,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM tenure_grouped
GROUP BY tenure_group
ORDER BY tenure_group;


/*
-------------------------------------------------------------------------------
4. Churn rate by preferred order category
-------------------------------------------------------------------------------
Business question:
Which product categories have the highest churn rate?
*/

SELECT
    preferred_order_cat,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM analytics_gold.gold_churn_model_input
GROUP BY preferred_order_cat
ORDER BY churn_rate_percent DESC;


/*
-------------------------------------------------------------------------------
5. Churn rate by preferred payment mode
-------------------------------------------------------------------------------
Business question:
Are some payment methods associated with higher churn?
Note:
Some categorical values may have the same meaning, such as:
- cod and cash on delivery
- cc and credit card
This query standardizes those values before analysis.
*/

WITH payment_standardized AS (
    SELECT
        customer_id,
        churn,
        CASE
            WHEN preferred_payment_mode = 'cod' THEN 'cash on delivery'
            WHEN preferred_payment_mode = 'cc' THEN 'credit card'
            ELSE preferred_payment_mode
        END AS payment_mode_standardized
    FROM analytics_gold.gold_churn_model_input
)

SELECT
    payment_mode_standardized,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM payment_standardized
GROUP BY payment_mode_standardized
ORDER BY churn_rate_percent DESC;


/*
-------------------------------------------------------------------------------
6. Churn rate by new customer flag
-------------------------------------------------------------------------------
Business question:
Do new customers churn more often?
*/

SELECT
    is_new_customer,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM analytics_gold.gold_churn_model_input
GROUP BY is_new_customer
ORDER BY is_new_customer;


/*
-------------------------------------------------------------------------------
7. Churn rate by high cashback flag
-------------------------------------------------------------------------------
Business question:
Are customers with higher cashback less likely to churn?
*/

SELECT
    high_cashback_customer_flag,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent,
    ROUND(AVG(cashback_amount)::numeric, 2) AS avg_cashback_amount
FROM analytics_gold.gold_churn_model_input
GROUP BY high_cashback_customer_flag
ORDER BY high_cashback_customer_flag;


/*
-------------------------------------------------------------------------------
8. Average customer behavior by churn status
-------------------------------------------------------------------------------
Business question:
How do churned and non-churned customers differ on key numeric variables?
*/

SELECT
    churn,
    COUNT(*) AS total_customers,
    ROUND(AVG(tenure)::numeric, 2) AS avg_tenure,
    ROUND(AVG(cashback_amount)::numeric, 2) AS avg_cashback_amount,
    ROUND(AVG(warehouse_to_home)::numeric, 2) AS avg_warehouse_to_home,
    ROUND(AVG(day_since_last_order)::numeric, 2) AS avg_day_since_last_order,
    ROUND(AVG(order_count)::numeric, 2) AS avg_order_count,
    ROUND(AVG(coupon_used)::numeric, 2) AS avg_coupon_used,
    ROUND(AVG(satisfaction_score)::numeric, 2) AS avg_satisfaction_score
FROM analytics_gold.gold_churn_model_input
GROUP BY churn
ORDER BY churn;


/*
-------------------------------------------------------------------------------
9. High-priority churn segment: new customers with complaints
-------------------------------------------------------------------------------
Business question:
Which customer segment should be prioritized first for retention?
*/

SELECT
    customer_id,
    tenure,
    complain,
    preferred_order_cat,
    preferred_payment_mode,
    cashback_amount,
    satisfaction_score,
    churn
FROM analytics_gold.gold_churn_model_input
WHERE is_new_customer = 1
  AND complain = 1
ORDER BY tenure ASC, cashback_amount ASC;


/*
-------------------------------------------------------------------------------
10. Mobile-related churn segment
-------------------------------------------------------------------------------
Business question:
How many churned customers are concentrated in mobile-related categories?
*/

SELECT
    preferred_order_cat,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_percent
FROM analytics_gold.gold_churn_model_input
WHERE preferred_order_cat IN ('mobile', 'mobile phone', 'laptop & accessory')
GROUP BY preferred_order_cat
ORDER BY churn_rate_percent DESC;


/*
-------------------------------------------------------------------------------
11. Potential retention action list
-------------------------------------------------------------------------------
Business question:
Which customers may need priority retention actions based on business rules?

Logic:
- New customers
- Complaint history
- Mobile-related category
- Lower cashback exposure
*/

SELECT
    customer_id,
    tenure,
    complain,
    preferred_order_cat,
    preferred_payment_mode,
    cashback_amount,
    CASE
        WHEN complain = 1 AND tenure <= 3 THEN 'Priority 1 - complaint recovery and onboarding'
        WHEN tenure <= 3 THEN 'Priority 2 - new customer onboarding'
        WHEN preferred_order_cat IN ('mobile', 'mobile phone', 'laptop & accessory') THEN 'Priority 3 - mobile category cross-sell'
        WHEN cashback_amount < 160 THEN 'Priority 4 - targeted incentive'
        ELSE 'Monitor'
    END AS suggested_retention_action
FROM analytics_gold.gold_churn_model_input
WHERE churn = 1
ORDER BY
    CASE
        WHEN complain = 1 AND tenure <= 3 THEN 1
        WHEN tenure <= 3 THEN 2
        WHEN preferred_order_cat IN ('mobile', 'mobile phone', 'laptop & accessory') THEN 3
        WHEN cashback_amount < 160 THEN 4
        ELSE 5
    END,
    tenure ASC;
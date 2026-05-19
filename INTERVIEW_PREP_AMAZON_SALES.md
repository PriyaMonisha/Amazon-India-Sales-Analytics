# Amazon India Sales Analytics — Interview Preparation Guide

**Target companies:** HCL, TCS, Wipro, Infosys | Flipkart, Amazon India, Swiggy, Meesho, PhonePe | Google India, Microsoft India

---

## How to Use This File
- **Service companies (HCL/TCS/Wipro):** Focus on Sections 1–5 + 9. Concept + project walkthrough.
- **Product companies (Flipkart/Amazon/Swiggy/Meesho):** Focus on Sections 2–8. Deep system design + ML depth.
- **MNC (Google/Microsoft India):** Focus on Sections 4–8 + 12. Theory, scalability, trade-offs.
- Update this file after EVERY section — never defer to end.

---

## Section 1 — Project Introduction (HR + First Technical Round)

**Q1. Tell me about your Amazon India Sales Analytics project in 2 minutes.**

> *(Fill in after completing all sections with real numbers)*
> Amazon India Sales Analytics is a production-grade ML platform I built on 11 years (2015–2025) of transaction data — roughly 1.1 million rows. The project combines a data engineering layer (11 CSVs → PostgreSQL star schema, orchestrated by Airflow), a feature store (Feast + Redis for training-serving consistency), five ML models (churn prediction, sales forecasting, price recommendation, product recommendation, anomaly detection), and a full production serving stack: FastAPI with Prometheus instrumentation, KS drift monitoring via Evidently AI, four Grafana dashboards, and a Streamlit multi-page dashboard. Everything runs in a 15-service Docker Compose stack.

**Q2. Why Amazon India specifically? Why 11 years of data?**
> *(Fill in)*

**Q3. What business decisions does this system enable?**
> *(Fill in)*

---

## Section 2 — Data Engineering

**Q4. Why a star schema? What is the tradeoff vs snowflake schema?**
> *(Fill in after Section 1)*

**Q5. You have 1.1M rows. How did you handle the ETL performance?**
> *(Fill in — chunked inserts, upsert strategy, indexes)*

**Q6. Walk me through one data quality challenge you solved.**
> *(Fill in — pick the most interesting of the 10 cleaning challenges)*

**Q7. What is Great Expectations? Why did you use it?**
> *(Fill in — ge.from_pandas(), 7 expectations, fail-fast in Airflow)*

---

## Section 3 — Feature Store

**Q8. What is training-serving skew? How did your feature store prevent it?**
> *(Fill in after Section 3)*

**Q9. Why Feast over a custom Redis implementation?**
> *(Key answer: point-in-time correctness, feature registry, offline/online split)*

**Q10. What's the difference between customer_training_features and customer_prediction_outputs?**
> *(Fill in — circular leakage prevention, two FeatureViews)*

---

## Section 4 — Model Selection & Evaluation

**Q11. Why XGBoost for churn over Logistic Regression?**
> *(Fill in after Section 4)*

**Q12. How did you handle class imbalance in the churn model?**
> *(Fill in — scale_pos_weight, ROC-AUC + F1, threshold optimization from PR curve)*

**Q13. Why Prophet over ARIMA for Indian e-commerce sales forecasting?**
> *(Key answer: handles non-linear trend, multiplicative seasonality, festival holidays, built-in uncertainty intervals)*

**Q14. What is WMAPE and why did you use it over MAPE?**
> *(Key answer: MAPE explodes when denominator near zero — new subcategories in 2015 have near-zero history)*

**Q15. How did you construct churn labels? What is right-censoring?**
> *(Key answer: 90-day inactivity after reference date; right-censoring = customers near dataset end have incomplete observation windows — excluded them)*

---

## Section 5 — Model Evaluation & Thresholds

**Q16. How did you choose the churn probability threshold?**
> *(Fill in — precision-recall curve, business cost ratio: false negative cost >> false positive cost)*

**Q17. What is your churn model's actual ROC-AUC and F1?**
> *(Fill in with real numbers from training run)*

**Q18. How does SHAP help your business users?**
> *(Fill in — /predict/churn/{id}/explain endpoint, top_factors per customer)*

---

## Section 6 — Time Series & Forecasting

**Q19. Walk me through your Prophet setup for Indian e-commerce.**
> *(Fill in — custom Diwali holidays, multiplicative seasonality, monthly freq, WMAPE results)*

**Q20. How did you validate your forecast models?**
> *(Fill in — WMAPE < 25% threshold, train 2015–2023, eval on 2024)*

---

## Section 7 — MLOps & Pipeline

**Q21. How does your retraining pipeline work end-to-end?**
> *(Fill in — model_retraining_dag, MLflow comparison, threshold gate before promotion)*

**Q22. What is MLflow? How did you use it?**
> *(Fill in — experiment tracking, 3-way model comparison, optimal_threshold logged)*

**Q23. How does your Airflow pipeline handle failures?**
> *(Fill in — TriggerDagRunOperator, GE validation blocks downstream, individual try/except)*

---

## Section 8 — Monitoring & Drift Detection

**Q24. What is model drift? How do you detect it?**
> *(Fill in — Evidently AI DataDriftPreset, KS test, drift_score key, Prometheus Gauge export)*

**Q25. What happens when your KS statistic exceeds 0.10?**
> *(Fill in — DRIFT_ALERTS counter incremented, Grafana panel turns red, triggers model_retraining_dag)*

**Q26. Why Evidently over raw KS test?**
> *(Fill in — per-feature drift, HTML reports, richer than univariate KS)*

---

## Section 9 — Production & Architecture

**Q27. Explain your system architecture in 2 minutes.**
> *(Fill in — data flow: CSVs → MinIO → Airflow → PostgreSQL → Feast/Redis → FastAPI → Prometheus → Grafana → Streamlit)*

**Q28. What happens when Redis goes down during inference?**
> *(Key answer: all Redis methods return None with warning log, inference continues without feature store, graceful degradation)*

**Q29. Why MinIO instead of GCS?**
> *(Key answer: identical boto3 API, 100% free locally, same code works with GCS in cloud by changing endpoint)*

**Q30. How does service startup order work in your Docker Compose?**
> *(Key answer: condition: service_healthy requires explicit healthcheck blocks; 3 init services exit 0; minio-init creates buckets before MLflow starts)*

---

## Section 10 — SQL Questions (Live Coding)

**Q31. Write a query to find the top 5 customers by CLV per state.**
```sql
WITH customer_clv AS (
    SELECT
        c.customer_id,
        c.customer_state,
        c.customer_city,
        c.is_prime_member,
        SUM(f.final_amount_inr)           AS total_revenue,
        COUNT(DISTINCT f.transaction_id)  AS total_orders,
        MIN(f.order_date)                 AS first_purchase,
        MAX(f.order_date)                 AS last_purchase
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_state, c.customer_city, c.is_prime_member
),
ranked AS (
    SELECT *,
        ROUND(total_revenue::numeric, 2) AS clv_inr,
        RANK() OVER (PARTITION BY customer_state ORDER BY total_revenue DESC) AS state_rank
    FROM customer_clv
)
SELECT customer_state, customer_id, customer_city, is_prime_member,
       clv_inr, total_orders, first_purchase, last_purchase, state_rank
FROM ranked
WHERE state_rank <= 5
ORDER BY customer_state, state_rank;
```

**Q32. Monthly revenue by subcategory (last 12 months):**
```sql
SELECT
    DATE_TRUNC('month', f.order_date) AS revenue_month,
    p.subcategory,
    SUM(f.final_amount_inr)           AS monthly_revenue
FROM fact_transactions f
JOIN dim_products p ON f.product_id = p.product_id
WHERE f.order_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY 1, 2
ORDER BY 1, monthly_revenue DESC;
```

---

## Section 11 — Behavioral / STAR Format

**Q33. Describe the biggest technical challenge and how you solved it.**
> *(Fill in — Evidently result structure bug, Feast path resolution, Docker healthcheck gap, or your actual biggest challenge)*

**Q34. How did you ensure reproducibility?**
> *(Key points: RANDOM_STATE=42 everywhere, MLflow tracks all params, config.py single source, APP_ENV for environment isolation)*

---

## Section 12 — Advanced / MNC Questions

**Q35. How would you scale this to real-time streaming?**
> *(Fill in — Kafka producer for transactions, Redis consumer updating features in real-time, Lambda Architecture: batch Airflow + streaming Kafka)*

**Q36. How would you add a new ML model without redeploying the entire stack?**
> *(Key answer: add model file to artifacts/, update FastAPI loaders dict, rolling restart of FastAPI container — zero downtime)*

**Q37. What would you do differently with 6 more months?**
> *(Fill in — A/B testing framework, streaming pipeline, AutoML for model selection, multi-region PostgreSQL)*

---

## Section 13 — Quick-Fire Conceptual

**Q38.** What is training-serving skew?
> Feature values computed differently at training time vs serving time. Prevented by Feast's shared feature registry.

**Q39.** What is right-censoring in churn modeling?
> Customers near dataset end have incomplete observation windows — can't know if they churned in the next 90 days. Must exclude them from labeling.

**Q40.** What does `scale_pos_weight` do in XGBoost?
> Compensates for class imbalance by weighting positive class samples. Value = `negative_count / positive_count`.

**Q41.** Why `model_to_json` for Prophet instead of pickle?
> Prophet uses Stan internally — pickle fails across environments. `model_to_json` serializes to a portable JSON format.

**Q42.** What is the FP-Growth algorithm?
> Frequent Pattern Growth — efficient alternative to Apriori for market basket analysis. Mines frequent itemsets without generating candidate sets.

**Q43.** What is WMAPE?
> Weighted Mean Absolute Percentage Error: `Σ|actual - predicted| / Σ|actual|`. Handles zero denominators that cause MAPE to explode.

**Q44.** What is the Kolmogorov-Smirnov test?
> A non-parametric test comparing two distributions. KS statistic = maximum absolute difference between two CDFs. We alert when KS > 0.10.

---

## Metrics to Memorize (fill in after training runs)

| Fact | Value |
|---|---|
| Dataset years | 2015–2025 (11 years) |
| Total transactions | 1,119,886 rows |
| Unique customers | 354,815 |
| Unique products | 2,004 |
| Subcategories | *[check df.subcategory.nunique()]* |
| ML models | 5 (churn, forecast, pricing, recommendation, anomaly) |
| FAST_MODE sample | 50,000 rows, 10 Optuna trials, 3-fold CV |
| Churn threshold | *[fill after PR curve optimization]* |
| Churn ROC-AUC | *[fill after training]* |
| Churn F1 | *[fill after training]* |
| Forecast WMAPE | *[fill after training — per subcategory]* |
| Docker services | 15 (12 running + 3 init) |
| GE expectations | 7 (on fact_transactions) |
| Airflow DAGs | 5 |
| RANDOM_STATE | 42 |
| Drift threshold | KS > 0.10 |

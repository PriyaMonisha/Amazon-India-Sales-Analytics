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

> Amazon India Sales Analytics is a production-grade ML platform I built on 11 years (2015–2025) of e-commerce transaction data — 1,127,609 raw rows across 11 yearly CSV files. The project has four layers: First, a data engineering layer — an ETL pipeline that cleans 10 real data quality challenges, loads into a PostgreSQL star schema with 4 tables, and is orchestrated by Airflow. Second, a feature store built on Feast + Redis that serves pre-computed customer, product, and geographic features with sub-millisecond latency while guaranteeing training-serving consistency. Third, five ML models — churn prediction (XGBoost), sales forecasting (Prophet per subcategory), price recommendation (RandomForest), product recommendation (FP-Growth market basket), and anomaly detection (IsolationForest) — all tracked in MLflow. Fourth, a production serving stack: FastAPI with Prometheus metrics, Evidently AI drift detection, four Grafana dashboards, and a 6-page Streamlit dashboard. Everything runs in a 15-service Docker Compose stack including MinIO for artifact storage and Airflow for orchestration.

**Q2. Why Amazon India specifically? Why 11 years of data?**

> Amazon India is one of the world's most complex e-commerce markets — 28 states, 6 official languages, massive income diversity from Metro to Rural, and a unique payment evolution story (COD → UPI). Eleven years of data (2015–2025) captures the full arc: early e-commerce adoption, demonetization's impact on digital payments, COVID's demand shock, and post-COVID normalisation. Single-year data would miss all of this context. For ML models specifically, 11 years gives enough historical signal for Prophet to learn genuine seasonal patterns (Diwali, Republic Day) rather than overfitting to one or two festivals.

**Q3. What business decisions does this system enable?**

> Six concrete decisions: (1) **Inventory planning** — the Prophet forecast tells operations how much stock to procure per subcategory 8 weeks before festival season. (2) **Customer retention** — churn model identifies customers likely to stop buying; marketing sends personalised re-engagement offers. (3) **Pricing strategy** — price recommendation model outputs the optimal discount percentage that maximises revenue without eroding margin. (4) **Cross-selling** — FP-Growth association rules power "Customers who bought X also bought Y" recommendations. (5) **Quality control** — anomaly detection flags sudden spikes in return rates, signalling a product quality issue before it scales. (6) **Campaign planning** — EDA shows Diwali and Back-to-School are the top revenue festivals, informing budget allocation.

---

## Section 2 — Data Engineering

**Q4. Why a star schema? What is the tradeoff vs snowflake schema?**

> A star schema has one fact table (fact_transactions, ~1.1M rows) connected to denormalized dimension tables (dim_customers, dim_products, dim_time). Snowflake schema would normalize further — e.g., a separate dim_city table referenced by dim_customers. The tradeoff: star schema uses more storage (city is stored per customer row) but query joins are simpler and faster — one level of joins vs two or three. For analytics workloads where query speed matters more than write efficiency, star schema is standard. We also built composite indexes on (order_date, customer_id) to accelerate the most common groupby patterns.

**Q5. You have 1.1M rows. How did you handle the ETL performance?**

> Two strategies: First, chunked inserts — we write 10,000 rows per transaction instead of one giant commit. This prevents timeout and allows partial recovery. Second, we use upsert (`INSERT ... ON CONFLICT DO UPDATE`) instead of DELETE+INSERT — this lets us re-run the ETL pipeline safely without data loss if it crashes mid-way. The upsert strategy means the ETL is idempotent: running it twice gives the same result. We also created indexes *after* the initial bulk load (not before), because PostgreSQL indexes slow down bulk inserts significantly.

**Q6. Walk me through one data quality challenge you solved.**

> The price columns had three separate problems: the rupee symbol (₹), Indian number formatting with multiple commas (₹1,25,000 = 125,000), and string values like "Price on Request". A single `float(val)` call would fail on all of these. Our `_parse_price()` function strips the ₹ symbol with regex, removes all commas, then casts to float. "Price on Request" and empty strings return None, which we then impute with the subcategory median. The key insight: we never silently ignore the problem — we track `prices_imputed` in a timestamped cleaning log, so we know exactly how many rows needed imputation after every ETL run.

**Q7. What data validation framework did you use? Why?**

> I originally specified Great Expectations, but GE 0.18.19 has a hard dependency on `ipywidgets` → `jupyterlab-widgets`, which ships static files with 300+ character file paths that exceed Windows' MAX_PATH limit (260 chars) in our project directory structure. Rather than requiring users to enable Long Path support (a system-level change), I switched to **Pandera** — a pure Python DataFrame schema validation library with zero Jupyter dependencies. It installs in seconds and gives the same quality gates via a cleaner, type-annotated API:
>
> ```python
> Column("final_amount_inr", Check.in_range(1, 500_000))
> Column("customer_id", nullable=False)
> Check(lambda df: 1_000_000 <= len(df) <= 1_300_000)
> ```
>
> The placement is the same as GE would be: runs as a dedicated Airflow `validation_dag` after `data_cleaning_dag`, before `feature_engineering_dag`. If any of the 7 rules fail, `SchemaError` is raised, Airflow marks the task failed, and all downstream DAGs are blocked — the feature store and model retraining never run on corrupt data.
>
> **Follow-up: "What are the 7 rules?"**
> 1. `final_amount_inr` between ₹1–₹5,00,000 (catches paise-scale exports, test data)
> 2. `customer_rating` between 0–5 (nullable — 30% missing is expected and fine)
> 3. `delivery_days` between 0–30 (catches "Same Day" text not cleaned, -3 day bugs)
> 4. `payment_method` in known set (catches new payment method not mapped in transform)
> 5. `transaction_id` unique (catches upsert bugs, file processed twice)
> 6. `customer_id` not null (catches upstream system anonymization errors)
> 7. Row count between 1M–1.3M (catches accidentally loading only 1 year, truncation)

**Q8. Why is idempotent ETL important? What does "upsert" mean?**

> Idempotent means: running the same operation twice produces the same result. In ETL, if the pipeline crashes at row 500,000 and restarts, idempotency ensures you don't end up with duplicate data or missing data — you just reload and the database ends up in exactly the same state. We achieve this with PostgreSQL's `INSERT ... ON CONFLICT (transaction_id) DO UPDATE SET ...` — if a row already exists with the same transaction_id, it's updated rather than duplicated. This is critical when Airflow retries a failed task automatically.

**Q9. Why did you not put a FK constraint on fact_transactions.order_date → dim_time?**

> FK constraints on fact tables in high-volume ETL cause two problems: (1) The constraint is checked on every insert — at 1.1M rows, this adds significant overhead. (2) Rows with invalid dates (NaT after failed parsing) would fail the insert entirely, causing the whole load to fail rather than gracefully handling bad dates. Our approach: map NaT dates to a sentinel row ('1900-01-01', festival='Unknown') in dim_time, and validate date ranges with Pandera before the load. Application-layer validation is more flexible and faster than DB-enforced FK constraints on fact tables.

---

## Section 3 — Exploratory Data Analysis (EDA)

**Q10. Walk me through your EDA. What were the most important findings?**

> I built 21 analyses querying PostgreSQL via SQLAlchemy, saving all charts to artifacts/charts/ so Streamlit can load them dynamically. The five most important findings:
>
> 1. **Smartphones = 73% of revenue** — one subcategory dominates. Everything else is secondary. This shaped which categories to prioritise in forecasting and recommendations.
> 2. **UPI replaced COD as India's dominant payment method** — grew from near-zero in 2015 to dominant by 2023. The payment evolution chart is a proxy for India's digital payments revolution.
> 3. **Festival sale orders have LOWER average order value (INR 47K) than regular sales (INR 78K)** — counterintuitive, but festivals attract budget-conscious buyers who purchase discounted mid-range products, not premium purchases.
> 4. **Only 6% of customers are Champions; 38% are Hibernating** — massive opportunity for re-engagement campaigns targeting the 24% At Risk segment.
> 5. **Top 36% of products generate 80% of revenue** — not the classic 20%, but the Pareto principle still holds. Stock decisions should focus on these 720 products.

**Q11. What was the most counterintuitive finding in your EDA?**

> Festival sale average order value being lower than regular sales. You'd expect Diwali shoppers to splurge on premium items — but the data shows the opposite. Festival sales attract deal hunters who buy mid-range phones at 40-50% off. The total festival VOLUME is huge (Back to School generated 185M INR, Diwali 171M INR), but each individual order is smaller. The insight: festival campaigns should focus on moving high-volume, mid-range inventory, not trying to upsell premium products.

**Q12. Why did you use PostgreSQL for EDA instead of pandas on CSV?**

> At 1.1M rows, pandas aggregations take 3-5 seconds per query. PostgreSQL with indexes (on order_date, customer_id, subcategory) runs the same GROUP BY in under 200ms. More importantly, having all analyses query the same database ensures consistency — if the ETL pipeline updates data, every chart automatically reflects the latest state on the next notebook run. It also demonstrates the full data engineering → analytics pipeline rather than just loading a CSV.

**Q13. How did you handle the seasonal patterns in your EDA?**

> Two approaches: First, the Monthly Revenue Heatmap (Year × Month grid, colour = revenue) makes seasonal patterns immediately visible — December is always the darkest cell, confirming Q4 as peak season. Second, I computed year-over-year growth percentages on the revenue trend and annotated each year's point. The 2020 COVID spike is clearly visible (+33% vs 2019), as is the post-2020 normalisation. For the forecasting models, this seasonality data was used to configure Prophet's holiday parameters — custom Diwali dates with a ±7-day window.

**Q14. What were your EDA technical choices and why?**

> Three key choices: (1) **All matplotlib/seaborn, no Plotly for static charts** — Plotly's `write_image()` uses Kaleido which launches a Chrome browser instance per chart (5 seconds each). With 21 charts, that's 100+ seconds of browser startup. Matplotlib saves to PNG in under 1 second. (2) **Discrete Set2 colour palette everywhere** — never sequential palettes on categorical data (they imply ordering where none exists). (3) **`showfliers=False` on boxplots** — keeping extreme outliers (1.2M INR orders) on the boxplot made the Y-axis 10x taller and hid the IQR where 95% of data lives. Hiding outliers with a note is more honest than distorting the chart scale.

**Q15. How did you handle the RFM segmentation technically?**

> RFM scores each customer 1-5 on Recency (time since last purchase), Frequency (number of orders), and Monetary value (total spend). The challenge was `pd.qcut` failing when there aren't enough distinct values to create 5 bins — common in FAST_MODE with a 50K sample where many customers appear only once. I wrote a `safe_qcut()` function that falls back to `pd.cut()` on rank-ordered values when qcut fails. The segments are then rule-based: Champions = R≥4 AND F≥4 AND M≥4; At Risk = R≤2 AND F≥3; etc. This is explainable to non-technical stakeholders — no black box.

---

## Section 4 — Feature Store

**Q16. What is training-serving skew? How did your feature store prevent it?**
> *(Fill in after Section 3 is built)*

**Q17. Why Feast over a custom Redis implementation?**
> *(Key answer: point-in-time correctness, feature registry, offline/online split)*

**Q18. What's the difference between customer_training_features and customer_prediction_outputs?**
> *(Fill in — circular leakage prevention, two FeatureViews)*

---

## Section 5 — Model Selection & Evaluation

**Q19. Why XGBoost for churn over Logistic Regression?**
> *(Fill in after Section 4 — ML models built)*

**Q20. How did you handle class imbalance in the churn model?**
> *(Fill in — scale_pos_weight, ROC-AUC + F1, threshold optimization from PR curve)*

**Q21. Why Prophet over ARIMA for Indian e-commerce sales forecasting?**
> *(Key answer: handles non-linear trend, multiplicative seasonality, festival holidays, built-in uncertainty intervals)*

**Q22. What is WMAPE and why did you use it over MAPE?**
> *(Key answer: MAPE explodes when denominator near zero — new subcategories in 2015 have near-zero history)*

**Q23. How did you construct churn labels? What is right-censoring?**
> *(Key answer: 90-day inactivity after reference date; right-censoring = customers near dataset end have incomplete observation windows — excluded them)*

---

## Section 6 — Model Evaluation & Thresholds

**Q24. How did you choose the churn probability threshold?**
> *(Fill in — precision-recall curve, business cost ratio: false negative cost >> false positive cost)*

**Q25. What is your churn model's actual ROC-AUC and F1?**
> *(Fill in with real numbers from training run)*

**Q26. How does SHAP help your business users?**
> *(Fill in — /predict/churn/{id}/explain endpoint, top_factors per customer)*

---

## Section 7 — Time Series & Forecasting

**Q27. Walk me through your Prophet setup for Indian e-commerce.**
> *(Fill in — custom Diwali holidays, multiplicative seasonality, monthly freq, WMAPE results)*

**Q28. How did you validate your forecast models?**
> *(Fill in — WMAPE < 25% threshold, train 2015–2023, eval on 2024)*

---

## Section 8 — MLOps & Pipeline

**Q29. How does your retraining pipeline work end-to-end?**
> *(Fill in — model_retraining_dag, MLflow comparison, threshold gate before promotion)*

**Q30. What is MLflow? How did you use it?**
> *(Fill in — experiment tracking, 3-way model comparison, optimal_threshold logged)*

**Q31. How does your Airflow pipeline handle failures?**
> *(Fill in — TriggerDagRunOperator, validation blocks downstream, individual try/except)*

---

## Section 9 — Monitoring & Drift Detection

**Q32. What is model drift? How do you detect it?**
> *(Fill in — Evidently AI DataDriftPreset, KS test, drift_score key, Prometheus Gauge export)*

**Q33. What happens when your KS statistic exceeds 0.10?**
> *(Fill in — DRIFT_ALERTS counter incremented, Grafana panel turns red, triggers model_retraining_dag)*

**Q34. Why Evidently over raw KS test?**
> *(Fill in — per-feature drift, HTML reports, richer than univariate KS)*

---

## Section 10 — Production & Architecture

**Q27. Explain your system architecture in 2 minutes.**
> *(Fill in — data flow: CSVs → MinIO → Airflow → PostgreSQL → Feast/Redis → FastAPI → Prometheus → Grafana → Streamlit)*

**Q28. What happens when Redis goes down during inference?**
> *(Key answer: all Redis methods return None with warning log, inference continues without feature store, graceful degradation)*

**Q29. Why MinIO instead of GCS?**
> *(Key answer: identical boto3 API, 100% free locally, same code works with GCS in cloud by changing endpoint)*

**Q30. How does service startup order work in your Docker Compose?**
> *(Key answer: condition: service_healthy requires explicit healthcheck blocks; 3 init services exit 0; minio-init creates buckets before MLflow starts)*

---

## Section 11 — SQL Questions (Live Coding)

**Q35. Write a query to find the top 5 customers by CLV per state.**
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

**Q36. Monthly revenue by subcategory (last 12 months):**
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

## Section 12 — Behavioral / STAR Format

**Q37. Describe the biggest technical challenge and how you solved it.**

> The biggest challenge was that Plotly's `fig.write_image()` for saving charts launched a full Chrome browser via Kaleido for each chart — 3-5 seconds per chart. With 21 charts, the EDA notebook took 100+ seconds just for chart export, and on some machines Kaleido crashed mid-run leaving corrupt PNG files. I diagnosed the root cause, then completely rewrote the EDA notebook using matplotlib/seaborn instead of Plotly for all static charts. The result: 21 charts generated in 41 seconds total with zero browser dependency. The tradeoff (less interactive charts) was acceptable since Streamlit displays static images anyway. This is the kind of practical engineering decision that matters in production — choose the right tool for the job, not the flashiest one.

**Q38. How did you handle a counterintuitive finding in your data?**

> The festival sale average order value being LOWER than regular sales (INR 47K vs INR 78K). My first instinct was "this must be a data quality bug." I investigated: verified the query was correct, checked the sample sizes (34K regular vs 15K festival orders), and cross-checked against the individual festival revenue chart. The finding held — it's real. The explanation: festivals attract budget-conscious buyers who wait for discounts on mid-range products (15K-30K phones). Regular periods have premium full-price purchases by less price-sensitive customers. I documented this in EDA_INSIGHTS.md with the business interpretation, so the marketing team doesn't make the wrong assumption that festival = premium customer.

**Q39. How did you ensure reproducibility?**

> Four practices: (1) `RANDOM_STATE = 42` defined once in config.py and passed everywhere — never hardcoded inline. (2) MLflow logs every training run's exact parameters, metrics, and git commit hash — any run from 6 months ago can be reproduced by reading its MLflow entry. (3) `config.py` with `APP_ENV` enum — local vs Docker vs Airflow all resolve to the correct database URL without any code changes. (4) Pinned dependency versions in requirements/*.txt — `pandera==0.31.1`, `xgboost==2.0.3`, etc. — so the environment is bit-reproducible.

---

## Section 13 — Advanced / MNC Questions

**Q40. How would you scale this to real-time streaming?**
> Add Kafka as the event bus: each transaction triggers a Kafka event → consumer updates Redis feature store in real-time instead of waiting for the nightly Airflow batch. The model serving layer (FastAPI) stays identical — it reads from Redis regardless of whether features came from batch or stream. This is the Lambda Architecture pattern: batch pipeline handles historical correctness, streaming handles recency. The churn model would benefit most — detecting a customer who just stopped buying patterns in real-time vs 24 hours later.

**Q41. How would you add a new ML model without redeploying the entire stack?**
> The FastAPI lifespan loads models from `artifacts/models/` at startup using a loaders dict. To add a new model: (1) Train it, save the artifact. (2) Add a loader function in `src/models/`. (3) Add the entry to the loaders dict in `api/main.py`. (4) Add the endpoint in `api/routers/predictions.py`. (5) Rolling restart of the FastAPI container — zero downtime since the old container handles requests until the new one is healthy. No other services need touching.

**Q42. What would you do differently with 6 more months?**
> Three things: (1) **A/B testing framework** — currently one churn model serves everyone. I'd add a feature flag system so 20% of traffic gets the challenger model and 80% gets the champion, with automatic promotion if challenger ROC-AUC > champion for 2 weeks. (2) **Real-time feature streaming** — replace the daily Airflow batch for feature computation with Kafka + Redis streams, reducing feature staleness from 24 hours to seconds. (3) **Model explainability dashboard** — SHAP values are computed per request, but surfacing them in a Streamlit admin panel with customer-level explanations would make the system genuinely useful for the retention team, not just the data science team.

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

## Metrics to Memorize

| Fact | Value |
|---|---|
| Dataset years | 2015–2025 (11 years) |
| Raw transactions | 1,127,609 rows |
| After cleaning | 1,122,000 rows (5,609 duplicates removed) |
| Unique customers | ~354,000 |
| Unique products | 2,004 |
| Subcategories | 6 (Smartphones, Laptops, Tablets, Smart Watch, Audio, TV & Entertainment) |
| Star schema tables | 4 (fact_transactions, dim_customers, dim_products, dim_time) |
| Pandera validation rules | 7 (on fact_transactions) |
| ML models | 5 (churn XGBoost, forecast Prophet, pricing RF, recommendation FP-Growth, anomaly IsolationForest) |
| FAST_MODE sample | 50,000 rows, 10 Optuna trials, 3-fold CV |
| EDA charts | 21 PNG files in artifacts/charts/ |
| Smartphones revenue share | 73.2% of total |
| Top state by revenue | Maharashtra |
| Prime vs Non-Prime AOV | INR 78K vs INR 62K (+25%) |
| Festival vs Regular AOV | INR 47K vs INR 78K (festival LOWER — budget buyers) |
| Top festival by revenue | Back to School (185M INR) |
| Highest return rate subcategory | Audio (8.1%) |
| Top 36% products | = 80% of revenue (Pareto) |
| Optimal discount range | 20-30% (highest revenue) |
| CLV Mean / Median | INR 74K / INR 47K |
| Delivery days (avg) | ~3.3-3.5 days (Metro to Rural) |
| Docker PostgreSQL port | 5433 (5432 occupied by local PostgreSQL) |
| Docker services | 15 (12 running + 3 init) |
| Airflow DAGs | 5 |
| RANDOM_STATE | 42 |
| Drift threshold | KS > 0.10 |
| Churn threshold | *(fill after PR curve optimization — Section 4)* |
| Churn ROC-AUC | *(fill after training — Section 4)* |
| Forecast WMAPE | *(fill after training — Section 4, per subcategory)* |

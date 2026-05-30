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
> Training-serving skew happens when the features your model sees at training time differ from what it sees at inference time — due to different code paths, different time windows, or stale data. In my platform, I prevented it three ways: (1) `CUSTOMER_TRAINING_FEATURE_REFS` is a single module-level list used in both `get_training_dataset()` and `get_online_features()` — identical feature set, identical names. (2) All SQL time windows (`total_orders_90d`, `avg_order_value_last_6m`) are computed relative to a `reference_date` parameter, never `CURRENT_DATE`, so the computation is reproducible at any point in time. (3) Feast's point-in-time join ensures that when I retrieve historical features for training, I only use data that existed before the label's observation date — no future leakage.

**Q17. Why Feast over a custom Redis implementation?**
> A custom Redis implementation solves the online serving problem but misses three things that Feast provides: (1) **Point-in-time correctness** — Feast's `get_historical_features()` joins entity rows with feature rows using `event_timestamp`, guaranteeing no future data leaks into training. A custom solution would require me to implement this join logic manually. (2) **Feature registry** — Feast maintains a versioned registry (`feast_registry.db`) so every feature has a documented schema, source, and TTL. Teams can discover what features exist and when they expire. (3) **Offline/online separation** — Feast manages the two-tier store (Parquet offline for batch training, Redis online for low-latency inference) with a single `materialize()` call, with no custom sync code needed. I used a file-based offline store (Parquet) rather than a PostgreSQL offline store to avoid the `feast[postgres]` package and keep dependencies minimal.

**Q18. What's the difference between customer_training_features and customer_prediction_outputs?**
> I created two completely separate FeatureViews to prevent circular leakage. `customer_training_features` holds 14 safe model inputs computed strictly from historical transaction data before the reference date: RFM scores, days since last purchase, order counts, average order values, prime membership, preferred category, return rate. These are computed by `compute_customer_features()` which queries PostgreSQL with a parameterized `reference_date`. `customer_prediction_outputs` holds 4 model output fields: `churn_probability`, `churn_label`, `clv_predicted`, `rfm_cluster`. These are written to Redis by the inference pipeline *after* the model runs. Critically, these outputs are **never read back as training inputs** — doing so would create circular dependency (training on the model's own previous predictions). The separation is enforced at the registry level: `CUSTOMER_TRAINING_FEATURE_REFS` only references `customer_training_features:*` fields.

**Q19. Why did you choose rank-based RFM scoring instead of quantile binning?**
> I originally planned to use `pd.qcut` for quintile binning, but Amazon India's transaction data is heavily right-skewed — roughly 40% of customers have exactly 1–2 orders. `pd.qcut` requires unique quantile boundaries, and with so many customers at the same value, multiple values fall on the same boundary, causing a `ValueError: Bin edges must be unique`. Instead I used `df["col"].rank(pct=True)` which assigns unique percentile ranks regardless of ties, then `pd.cut` with fixed bins `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` to map percentiles to quintile scores 1–5. This always produces clean bins regardless of distribution skewness.

**Q20. How do you define your RFM customer segments?**
> I used explicit rule-based thresholds on the 1–5 quintile scores. Champions (R≥4, F≥4, M≥4) are the top 20% on all three dimensions — recent, frequent, high spend. At Risk customers (R≤2, F≥3, M≥3) are historically good customers who've stopped buying recently — they get priority retention offers because the cost of losing a high-LTV customer is much higher than a false positive. New customers (R≥4, F≤2) bought recently but haven't established a frequency pattern yet. Lost customers (R≤2, F≤2, M≤2) are low on all dimensions. I chose rule-based over clustering because the segments are interpretable to the business team and directly actionable — "send a winback offer to At Risk" is a concrete campaign, whereas a cluster label is not.

---

## Section 5 — Model Selection & Evaluation

**Q19. Why XGBoost for churn over Logistic Regression?**
> XGBoost handles non-linear feature interactions (e.g., RFM score interactions, recency × frequency) that logistic regression misses. It natively handles missing values (e.g., customers with no 6-month history → NaN avg_order_value_last_6m). With `scale_pos_weight` for class imbalance and Optuna hyperparameter search with cross-validation, it achieves significantly higher ROC-AUC than logistic regression on this dataset. It also produces SHAP values for per-customer explainability — critical for business stakeholders.

**Q20. How did you handle class imbalance in the churn model?**
> Three techniques combined: (1) `scale_pos_weight = count(non-churned) / count(churned)` — tells XGBoost to penalise missing churned customers more heavily. (2) ROC-AUC as the primary metric (insensitive to class ratio). (3) F1-optimal threshold from precision-recall curve — never using default 0.5. False negatives (missed churners) cost more than false positives (wrong retention offers), so we bias recall.

**Q21. Why Prophet over ARIMA for Indian e-commerce sales forecasting?**
> Prophet handles three critical requirements: (1) non-linear growth trend (Indian e-commerce grew 40×+ from 2015 to 2025), (2) multiplicative seasonality — Diwali spikes are 3–5× baseline, not additive, (3) custom holiday effects — we added Diwali dates with a -7/+3 day window that ARIMA cannot model. Prophet also provides uncertainty intervals out-of-the-box and handles missing months gracefully.

**Q22. What is WMAPE and why did you use it over MAPE?**
> WMAPE = Σ|actual - forecast| / Σ|actual|. MAPE = mean(|actual - forecast| / actual) which explodes to infinity when actual ≈ 0 — e.g., a new Electronics subcategory launched in 2024 with ₹500 revenue in month 1. WMAPE weights errors by actual magnitude, so small-volume subcategories don't dominate the metric. Our threshold is WMAPE < 0.25 (25%).

**Q23. How did you construct churn labels? What is right-censoring?**
> Churn = customer made no purchase in the 90 days after a reference date. Reference dates: 2023-01-01 (train set, label = no purchase Jan–Mar 2023) and 2024-01-01 (test set, label = no purchase Jan–Mar 2024). Right-censoring: if a customer's observation window extends past the dataset end date (2025-12-31), we cannot observe whether they churned — their label would be wrong. We guard against this with: `if obs_end > DATASET_END_DATE: raise ValueError`. Both reference dates pass (obs_end = 2023-04-01 and 2024-04-01, both before 2025-12-31).

---

## Section 6 — Model Evaluation & Thresholds

**Q24. How did you choose the churn probability threshold?**
> F1-maximising threshold from the precision-recall curve on the 2024 test cohort. We compute precision and recall at every threshold point, calculate F1 = 2PR/(P+R), and select the threshold with the highest F1. This reflects the business cost ratio: missing a churner (false negative) costs more than sending a retention offer to a loyal customer (false positive). The threshold is stored in `baseline_churn_proba.json` and loaded by FastAPI.

**Q25. What is your churn model's actual ROC-AUC and F1?**
> Requires running `notebooks/03_ml_training.py` with PostgreSQL. Minimum threshold is ROC-AUC ≥ 0.72 (from config.py). Expected range based on dataset characteristics: ROC-AUC 0.74–0.82, F1 0.55–0.70 (churn class). The model logs real numbers to MLflow — viewable at `http://localhost:5000`.

**Q26. How does SHAP help your business users?**
> SHAP TreeExplainer is pre-computed during training and pickled as `churn_explainer.pkl`. FastAPI's `/predict/churn/{customer_id}/explain` endpoint loads it from `app.state.models["churn"]["explainer"]` (loaded once at startup, never per-request). It returns the top 5 SHAP features for that specific customer: e.g., "days_since_last_purchase=127 pushed churn probability +0.23". This converts a black-box prediction into an actionable retention reason.

---

## Section 7 — Time Series & Forecasting

**Q27. Walk me through your Prophet setup for Indian e-commerce.**
> Monthly aggregation (freq='MS') from PostgreSQL via `date_trunc('month', order_date)`. Prophet config: `yearly_seasonality=True` (strong annual pattern), `weekly_seasonality=False` (monthly data has no weekly pattern), `daily_seasonality=False`, `seasonality_mode='multiplicative'` (festival spikes are proportional to trend, not absolute). Custom Diwali holidays DataFrame with `lower_window=-7, upper_window=3` — 7 pre-Diwali days of demand surge plus 3 days after. Never `add_country_holidays()` alone — that approximates Diwali poorly and double-counts with manual regressors.

**Q28. How did you validate your forecast models?**
> Last 3 months held out as test set per subcategory. Metric: WMAPE (robust to near-zero denominators). Minimum data: 12 months required for yearly seasonality — subcategories below this threshold get a warning and are skipped. Models logged to MLflow with WMAPE + n_train_months per subcategory. Slugs of all trained subcategories saved to `forecast_slugs.json` for FastAPI routing.

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

---

## Section 8: Streamlit Dashboard (Q45–Q50)

**Q45.** Why Streamlit for the analytics dashboard?
> Streamlit lets data scientists build interactive dashboards in pure Python — no frontend skills needed. We use Plotly for charts (interactive), SQLAlchemy for DB queries, and `requests` to call FastAPI. `@st.cache_data(ttl=300)` prevents hammering PostgreSQL on every rerender.

**Q46.** How does the Streamlit app interact with ML models?
> It never loads models directly. Instead it calls FastAPI endpoints (`/predict/churn/{id}`, `/predict/forecast/{subcat}`, etc.) via HTTP. This separates concerns: Streamlit handles UI, FastAPI handles inference. Both run as separate Docker services.

**Q47.** What does the churn page show when models aren't trained?
> Graceful degradation — it shows a `st.info()` message "Train models first" for charts that need `baseline_churn_proba.json`. The live predictor form still renders but returns a 503 error with a clear message. PostgreSQL-based charts (customer tier distribution, MAU trend) still load.

**Q48.** How does the demand forecast page work?
> User selects a subcategory → Streamlit calls `GET /predict/forecast/{subcat}?periods=6` → FastAPI returns 6 months of Prophet predictions with 95% confidence intervals. We render a line chart with a shaded CI band using Plotly. Historical sales come from PostgreSQL for context.

**Q49.** How does the pricing analytics page demonstrate price elasticity?
> User inputs a subcategory + current price → calls `GET /predict/pricing/{subcat}?current_price=X` → FastAPI returns 5 scenarios (−20%, −10%, 0%, +10%, +20%). We render: (1) revenue bar chart, (2) demand line chart, (3) bubble chart (price × revenue × demand size), (4) optimal price highlighted in green.

**Q50.** What makes the anomaly page interactive?
> User enters transaction details (final amount, MRP, delivery days, return flag) → POST `/predict/anomaly` → response includes `is_anomaly` flag and `anomaly_score` (IsolationForest decision function). We render a gauge chart centered at 0 (negative = anomalous), and overlay the user's discount/delivery values on historical distributions from PostgreSQL for context.

---

## Section 9: Airflow Orchestration (Q51–Q57)

**Q51.** Why use Airflow instead of a simple cron job?
> Cron gives you scheduling but nothing else — no retry logic, no dependency management, no UI for monitoring, no XCom for passing data between steps. Airflow gives us DAG visualization, per-task retries with configurable delay, TriggerDagRunOperator for cross-DAG chaining, and a searchable log history. For a pipeline with 4 chained stages (ETL→features→training→serving check), Airflow's dependency graph makes failures immediately diagnosable.

**Q52.** How do you chain 4 DAGs so they run sequentially?
> Using TriggerDagRunOperator with `wait_for_completion=False`. The final task of each DAG triggers the next DAG ID. We use `wait_for_completion=False` because the triggered DAG runs in a separate DagRun — waiting would block the triggering DAG's executor slot. The chain is: `etl_pipeline` → `feature_engineering` → `model_training` → `model_serving_check`. `drift_monitoring` runs independently on `@daily`.

**Q53.** Why not put all logic in one DAG instead of four?
> Separation of concerns and independent restartability. If feature engineering fails, we can restart just that DAG without re-running ETL. If model training fails, we can retrain without re-extracting. Each DAG can also be triggered independently for ad-hoc runs. Four small DAGs are also easier to monitor in the Airflow UI than one DAG with 15 tasks.

**Q54.** Why are all `src.*` imports inside task functions, not at module level?
> DAG module code runs in the Airflow scheduler process when it parses DAGs. Task callables run in worker processes. If we import `from src.models.churn import train_churn_model` at module level, the scheduler process needs all ML dependencies installed — including XGBoost, Prophet, SHAP. By moving imports inside callables, the scheduler only needs `airflow` installed; workers get the heavy imports. It also ensures `sys.path.insert` from `utils.py` fires before any `src.*` import.

**Q55.** How do you pass data between tasks without bloating the metadata DB?
> XCom stores values in Airflow's metadata database — fine for small strings, terrible for DataFrames. We store intermediate data as Parquet files in `PROCESSED_DATA_DIR` and push only the file path string via XCom. For example: `extract` saves `airflow_raw_sales.parquet`, pushes `"raw_sales_path"` → `transform` pulls the path with explicit `task_ids="extract"` and reads the Parquet. This keeps XCom lightweight and keeps large data in the filesystem.

**Q56.** Why does the drift DAG call FastAPI instead of running Evidently directly?
> The Airflow worker is a separate process from FastAPI. If we call `compute_churn_drift()` directly in the Airflow worker, Evidently updates the Prometheus gauges in the worker's process-local registry — a registry that Prometheus never scrapes. The Prometheus scraper only talks to FastAPI's `/metrics` endpoint. By calling `POST /monitor/drift/run` via HTTP, FastAPI's own process updates its own registry, and the drift scores appear in Grafana as expected.

**Q57.** After model training, why doesn't the serving check see the new models?
> FastAPI loads models at startup via the lifespan hook (`@asynccontextmanager`). When training writes new artifact files, the FastAPI container still has the old models in memory — it doesn't watch the filesystem for changes. The serving check is therefore only valid after `docker compose restart fastapi`. In production, you'd add a `/admin/reload` endpoint that re-runs the lifespan loading logic, or use a model registry (MLflow Model Registry, BentoML) that pushes change notifications to the serving layer.

---

## Section 10: Docker Compose (Q58–Q64)

**Q58.** Why use single-stage instead of multi-stage Dockerfiles?
> All dependencies — XGBoost 2.0.3, Prophet 1.1.5, scikit-learn 1.5.2 — ship as pre-built wheels on PyPI for Python 3.11 Linux x86_64. `pip install` downloads wheels; no compilation happens. Multi-stage builds are valuable when you compile C extensions from source and want to exclude the compiler toolchain from the final image. Here there's nothing to exclude. `libgomp1` for XGBoost's OpenMP thread pool belongs in the runtime image anyway. Multi-stage would add 20+ lines of Dockerfile complexity for zero image-size benefit.

**Q59.** Why does FastAPI have `mem_limit: 3g` but Streamlit only `1g`?
> FastAPI loads all Prophet models at startup in a `lifespan` hook. Prophet's in-memory footprint after JSON deserialization is ~150MB per model. With ~20 subcategory models: 20 × 150MB = 3GB for Prophet alone, plus XGBoost churn model (~50MB), IsolationForest (~100MB), and Python/uvicorn overhead (~200MB). Total: ~3.35GB minimum. Using `--workers 1` is critical — N uvicorn workers means N full copies of all models in memory; 2 workers would require 6GB+ and OOMKill the container. FastAPI's async event loop handles concurrency within a single worker. Streamlit calls FastAPI via HTTP and never loads ML libraries directly, so 1g is sufficient.

**Q60.** What is the init container pattern and why use it here?
> An init container runs to completion (exit 0) before dependent services start. `condition: service_completed_successfully` in `depends_on` blocks downstream services until init exits cleanly. We use it for two idempotent setup tasks: `minio-init` creates MinIO buckets (`mc mb --ignore-existing`), and `airflow-init` runs `airflow db init` and creates the admin user. The alternative — baking setup into the main service entrypoint — re-runs setup on every container restart, causing "user already exists" errors and broken states. Init containers are a clean separation: setup runs once, service runs forever.

**Q61.** What does `start_period` do in health checks, and why is it important?
> `start_period` is a grace period during which health check failures don't count toward `retries`. Without it, Docker fires the first probe immediately after container start. PostgreSQL takes ~20-30 seconds to finish initializing; FastAPI takes ~60 seconds to load all ML models. Without `start_period`, Docker marks them unhealthy on the first probe and blocks all `condition: service_healthy` dependent services from starting. With `start_period: 30s` on Postgres and `start_period: 60s` on FastAPI, Docker waits before counting failures — allowing legitimate startup time.

**Q62.** When do you use named volumes vs bind mounts?
> Named volumes (`postgres_data`, `redis_data`, `minio_data`, `grafana_data`) for persistent service state managed by Docker. They survive `docker compose down` and are deleted only with `docker compose down -v`. Bind mounts (`./dags`, `./feast_repo`, `./artifacts/models:ro`) for host-side files that need to be live-editable or produced by local processes. The `:ro` flag on `prometheus.yml` and model artifacts prevents the container from accidentally writing to host files and avoids permission errors on Linux. Never add `-v` to teardown unless intentionally resetting from scratch — it deletes all database data.

**Q63.** Why is `REDIS_CONNECTION_STRING: redis:6379` needed in the FastAPI service environment?
> `feast_repo/feature_store.yaml` configures the online store connection as `${REDIS_CONNECTION_STRING:-localhost:6379}`. Inside the FastAPI container, `localhost` refers to the container itself, not the Redis service. Docker Compose creates a bridge network where each service name is a DNS hostname — `redis` resolves to the Redis container's IP. Without setting `REDIS_CONNECTION_STRING: redis:6379` in docker-compose.yml, Feast would try to connect to `localhost:6379` inside the container and fail. The env var pattern (with a `:-localhost:6379` default) was already in place from Rule 29 — Docker just needs to override the default.

**Q64.** Why run as a non-root user in both Dockerfiles?
> Without a non-root user, any file the container writes at runtime — drift reports in `artifacts/drift/`, temporary files — is owned by `root` on the host bind mount. This causes `git status` to show unexpected changes and requires `sudo` for `rm` or `make clean`. The fix is `groupadd -r appuser && useradd -r -g appuser` plus `chown -R appuser:appuser /app` before switching to that user. It also follows the principle of least privilege: a compromised process running as root inside the container has broader escape potential than one running as an unprivileged user.

---

## Section 11 — Test Suite (pytest, mocking, fixtures)

**Q65.** Why mock all external services in the test suite instead of testing against real PostgreSQL and Redis?

> Speed and isolation. The full 95-test suite runs in under 90 seconds with mocks; the same suite against real services would need Docker running, database migrations, Feast materialization, and MLflow tracking — that's a 5-10 minute cold-start per run. More importantly, external services introduce flakiness: network failures, stale data, port conflicts. Mocked tests are deterministic by construction. We still have `@pytest.mark.integration` tests for ETL pipeline correctness that run against real PostgreSQL in CI — those test the DB interaction explicitly. The split is: unit/contract tests run everywhere (pytest, IDE, pre-commit), integration tests run in CI only against ephemeral Docker services.

**Q66.** What is the Prometheus duplicate-metric problem, and how does the test suite solve it?

> Metrics register at import time — when `api.main` is first imported, the counters and histograms are registered with the global `REGISTRY`. In IDE test runners (JetBrains, VS Code) the test process often persists across runs without a fresh interpreter. Re-importing `api.main` tries to re-register the same metric names and raises `ValueError: Duplicated timeseries in CollectorRegistry`. The fix is a session-scoped autouse fixture `_isolate_prometheus` that snapshots all registered collectors before the session (`_names_to_collectors.values() | _collectors_without_names`) and unregisters any new ones on teardown. This is a no-op in standard single-run pytest but guards interactive re-runs.

**Q67.** Why is `trained_anomaly_bundle` session-scoped while most other fixtures are function-scoped?

> IsolationForest training takes ~50ms on 100 rows — small but non-trivial when multiplied across 7 tests in `TestDetectAnomalies` plus the API fixtures that depend on `mock_anomaly_bundle`. Session scope trains once per pytest run and reuses the same model+scaler objects across all tests. The risk is shared mutable state, but `detect_anomalies()` is a pure read-only function (never mutates the model), so sharing is safe. Function scope for fixtures like `sample_sales_df` and `mock_pricing_result` prevents in-place mutation bugs — one test modifying a DataFrame or dict would contaminate subsequent tests if they shared the fixture instance.

**Q68.** Why use Starlette `TestClient` instead of `httpx.AsyncClient` for FastAPI tests?

> `TestClient` wraps the ASGI app synchronously using `requests` under the hood — no event loop, no `await`, no `pytest-asyncio` configuration needed. It works because all our FastAPI endpoints are defined as `def` (synchronous), not `async def`. Starlette's ASGI interface handles the event loop internally; the test code stays synchronous. `httpx.AsyncClient` would require `@pytest.mark.asyncio` on every test method and `asyncio_mode = auto` in pytest.ini — significant boilerplate for no benefit when the endpoints don't use async I/O.

**Q69.** What does `raise_server_exceptions=True` vs `False` do, and why does each test client use a different setting?

> `raise_server_exceptions=True` (used in `api_client_all_loaded`): server-side Python exceptions propagate as real exceptions in the test. If a router raises an unhandled `AttributeError`, pytest shows `ERROR` with the full traceback pointing at the bug — the most informative failure mode. `raise_server_exceptions=False` (used in `api_client_no_models`): `HTTPException` is caught and returned as an HTTP response with the appropriate status code. This setting is required for 503-tests — the router intentionally raises `HTTPException(503)` when models are None, and we want to assert `r.status_code == 503` rather than catch a Python exception.

**Q70.** How do you test model loading when no trained artifacts exist on disk?

> `monkeypatch.setattr("src.models.churn.ARTIFACTS_DIR", tmp_path)` redirects the module-level constant to a temp directory that has the directory structure (an empty `models/` subdirectory) but no artifact files. When `load_churn_model()` tries to open `churn_model.json`, it hits `FileNotFoundError` naturally — no artificial exception mocking needed. The monkeypatch is function-scoped and reverses automatically, so subsequent tests see the original `ARTIFACTS_DIR`. The key insight is patching the name in the module's own namespace (`src.models.churn.ARTIFACTS_DIR`) rather than in `config`, because Python binds the name at import time.

**Q71.** What's the difference between unit, integration, and contract tests in this ML pipeline?

> Unit tests (most of the suite) test a single function in isolation: `slug_from_subcategory("Smart TVs") == "smart_tvs"`, `_parse_price("₹1,299") == 1299.0`, `discount_pct` clipping at 0 and 1. Contract tests verify interface agreements between components without testing behavior: `ChurnModelBundle.__annotations__.keys()` matches the 8 expected keys, `len(ALL_FEATURES) == 14`, the `/health` response includes exactly the 5 expected model names. These fail if a team member renames a key or adds a feature without updating the contract. Integration tests (`@pytest.mark.integration`) test multi-component flows with real services — ETL pipeline against PostgreSQL, Feast materialization against Redis. They run in CI only, not on every local pytest run.

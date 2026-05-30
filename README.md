# Amazon India Sales Analytics — End-to-End ML Platform

> **11 years of Amazon India sales data (2015–2025) · 1.1 million transactions · 5 production ML models · Full MLOps stack**

![CI](https://github.com/PriyaMonisha/Amazon-India-Sales-Analytics/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat)
![Prophet](https://img.shields.io/badge/Prophet-1.1-blue?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE?style=flat&logo=apache-airflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=flat&logo=mlflow&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Prometheus-F46800?style=flat&logo=grafana&logoColor=white)

A production-grade machine learning platform built on a decade of Amazon India e-commerce data. Covers the complete ML lifecycle: raw data ingestion → feature engineering → model training → real-time serving → monitoring → interactive dashboard.

**Engineering highlights:** sub-millisecond inference latency · authenticated API with rate limiting · drift-triggered auto-retraining · hash-based A/B testing · SHAP feature importance monitoring · versioned model registry · 96 automated tests · GitHub Actions CI

---

## Documentation

| Document | Description | Regenerate |
|----------|-------------|------------|
| [Analytics Report (PDF)](artifacts/reports/Analytics_Report_Amazon_India_Sales_Analytics.pdf) | 23 EDA analyses with business insights, strategic recommendations, and executive summary | `make docs` |
| [Data Dictionary (PDF)](artifacts/reports/Data_Dictionary_Amazon_India_Sales_Analytics.pdf) | Full schema reference — all tables, columns, types, constraints, and business definitions | `make docs` |

> PDFs are pre-built and committed. To regenerate after code changes: `make docs`

---

## What This Project Does

| Layer | What's Built |
|-------|-------------|
| **Data Engineering** | ETL pipeline processing 11 yearly CSVs into a PostgreSQL star schema (1.1M rows, 4 tables, 10 cleaning challenges) |
| **Feature Store** | Feast + Redis for online/offline feature serving; 14 customer features; RFM segmentation |
| **ML Models** | 5 models: churn prediction, demand forecasting, price elasticity, product recommendations, anomaly detection |
| **Model Registry** | Versioned artifact registry (`src/model_registry/`) with atomic writes, promotion log, and one-command rollback |
| **A/B Testing** | Hash-based deterministic champion/challenger routing with nightly outcome materialization and auto-promotion DAG |
| **API** | FastAPI with 5 prediction endpoints + API key auth + rate limiting + `/health/models` registry endpoint |
| **Monitoring** | Evidently drift detection · Prometheus metrics · SHAP feature importance drift · 4 Grafana dashboards |
| **Auto-Retraining** | Airflow DAG that reads persisted drift results and triggers retraining only when drift exceeds threshold |
| **Dashboard** | Streamlit app with **10 pages and 50+ interactive charts** |
| **Orchestration** | 7 Airflow DAGs (5 ETL/inference + 2 MLOps: retraining + A/B promotion) |
| **Infra** | Full Docker Compose stack (15 services) with health checks and volume mounts |
| **Tests** | 96 automated tests across ETL, API, and model layers · GitHub Actions CI on every push |

---

## Architecture

```
Raw CSVs (11 years, 1.1M rows)
         │
         ▼
  ETL Pipeline ──────────────────────► PostgreSQL (star schema)
  (extract / transform / validate)          │
                                            │
                               ┌────────────┴───────────┐
                               ▼                        ▼
                       Feature Store             EDA (23 charts)
                     (Feast + Redis)
                               │
                     ┌─────────┴──────────┐
                     ▼                    ▼
               Offline Store        Online Store
              (Parquet files)       (Redis cache)
                     │
                     ▼
             ML Training (5 models)
          ┌──────┬──────┬──────┬──────┐
          ▼      ▼      ▼      ▼      ▼
        Churn  Forecast Price  Reco  Anomaly
       XGBoost Prophet  XGB  FP-Growth  IsoForest
          └──────┴──────┴──────┴──────┘
                     │
                     ▼
               FastAPI (5 routers)
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
     Streamlit Dashboard   Prometheus + Grafana
      (10 pages, 50+ charts) (4 dashboards, drift alerts)
                     │
                     ▼
             Airflow Orchestration
           (5 DAGs, weekly ETL → daily drift)
```

---

## Dashboard Pages (10 Pages)

| Page | Analytics Coverage | Key Charts |
|------|--------------------|------------|
| **0 — Executive Dashboard** | KPI monitoring, strategic overview, financial performance, growth analytics | KPI alerts, YoY growth, market share, financial performance, BI command centre |
| **1 — Customer Churn** | Churn prediction, customer segmentation | XGBoost probability distribution, confusion matrix, SHAP waterfall, live predictor |
| **2 — Demand Forecast** | Sales forecasting, seasonal trends | Prophet forecast + CI bands, YoY growth, seasonal heatmap |
| **3 — Pricing Analytics** | Price optimization, demand sensitivity | Revenue scenarios, demand sensitivity, price-revenue bubble chart |
| **4 — Recommendations** | Product recommendations, basket analysis | Association rules, confidence-lift scatter, co-purchase heatmap |
| **5 — Anomaly Detection** | Revenue + return anomalies | Score gauge, distribution comparisons, live transaction checker |
| **6 — Festival & Seasonal** | Festival impact, seasonality patterns | Festival revenue comparison, festival vs regular by year, seasonal pattern, subcategory breakdown |
| **7 — Prime & Demographics** | Prime member analysis, demographics | Prime KPI cards, category preferences, age group spending, tier spending profile |
| **8 — Brand & Products** | Brand performance, product lifecycle | Brand market share, product ratings, new launch performance, return analysis |
| **9 — Customer Journey** | Purchase frequency, CLV, journey mapping | Purchase frequency, category transitions, CLV by segment, product scorecard |

---

## ML Models

### 1. Customer Churn Prediction
- **Algorithm:** XGBoost with Optuna hyperparameter tuning
- **Label:** No purchase in next 90 days (temporal definition — not return status)
- **Split:** Temporal — train on 2015–2023, test on 2024 cohort
- **Features:** 14 RFM + behavioral features from Feast feature store
- **Explainability:** SHAP values served per prediction via FastAPI

### 2. Demand Forecasting (Prophet)
- **Algorithm:** Facebook Prophet per product subcategory
- **Seasonality:** Multiplicative; custom Diwali/festival regressors
- **Granularity:** Monthly per subcategory; 12-month horizon
- **Serialization:** `model_to_json` (portable across environments)

### 3. Price Elasticity
- **Algorithm:** XGBoost log-log regression (price → demand)
- **Output:** Elasticity coefficient + revenue-maximising price per SKU

### 4. Product Recommendations (FP-Growth)
- **Algorithm:** mlxtend FP-Growth on order-level baskets
- **Fallback:** Progressive min_support + popularity fallback

### 5. Anomaly Detection
- **Algorithm:** IsolationForest on transaction features
- **Output:** Anomaly score + binary flag via FastAPI

---

## Tech Stack

```
Python 3.11        XGBoost 2.0        Prophet 1.1.5
scikit-learn 1.5   mlxtend 0.23       Optuna 3.x
MLflow 2.14        Feast 0.38         Redis 5.0
FastAPI 0.115      Streamlit 1.35     Evidently 0.4
Airflow 2.8        PostgreSQL 15/18   Prometheus + Grafana
Docker Compose     Pandera            SHAP
```

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_cleaning.py       # 10 cleaning challenges on raw CSVs
│   ├── 01_data_engineering.py    # ETL → PostgreSQL star schema
│   ├── 02_eda.py                 # 23 EDA charts saved to artifacts/charts/
│   └── 03_ml_training.py         # All 5 models + MLflow logging
├── src/
│   ├── etl/                      # extract.py, transform.py, load.py
│   ├── features/                 # compute.py (RFM), feature_store.py
│   ├── models/                   # churn, forecasting, pricing, recommendation, anomaly
│   ├── model_registry/           # registry.py — versioned artifact registry with atomic writes
│   ├── monitoring/               # metrics.py (Prometheus), drift.py (Evidently), shap_monitoring.py
│   ├── ab_testing/               # router.py (hash routing), tracker.py (AUC comparison), ddl.py
│   ├── utils/                    # mlflow_utils.py (shared setup)
│   └── validation/               # expectations.py (Pandera, 7 schema checks)
├── api/
│   ├── main.py                   # FastAPI with lifespan model loading + A/B challenger
│   ├── dependencies.py           # Shared auth (API key) + rate limiter (no circular imports)
│   ├── models.py                 # Pydantic request/response schemas
│   └── routers/                  # churn (with A/B routing), forecast, pricing, recommendation, anomaly
├── streamlit_app/
│   ├── app.py                    # Overview page (KPIs, trends, geography)
│   ├── pages/                    # 9 additional pages (0–9)
│   └── utils/                    # db.py (queries), api_client.py, offline.py
├── dags/                         # 7 Airflow DAGs (5 ETL/inference + model_retraining + ab_promotion)
├── scripts/                      # train_churn.py (CLI wrapper), materialize_ab_outcomes.py
├── feast_repo/                   # Feature views, entities, feature services
├── monitoring/                   # Grafana dashboards, Prometheus config
├── .github/workflows/            # ci.yml — lint + 96 tests on every push
├── tests/                        # test_etl.py, test_api.py, test_models.py (96 tests)
├── requirements/                 # base.txt, dev.txt, api.txt, ml.txt, etl.txt
├── Dockerfile.api
├── Dockerfile.streamlit
└── docker-compose.yml            # 15-service full stack
```

---

## Quick Start

### Fastest way to verify the code (no data or database needed)

```bash
git clone https://github.com/PriyaMonisha/Amazon-India-Sales-Analytics.git
cd Amazon-India-Sales-Analytics
make setup     # creates venv + installs deps + copies .env.example → .env
make test      # runs all 96 tests — uses mocks, no database or data files required
```

All 96 tests pass without any external setup. Tests cover ETL transformations, all 5 API prediction endpoints, and model training logic — using in-memory fixtures and mocks.

> **Note on data files:** Raw CSVs (1.1 M rows) and trained model artifacts are not committed to the repo due to size. Running the full pipeline (`make etl` → `make train` → `make serve`) requires PostgreSQL and the source data files. The test suite demonstrates code correctness without them.

---

### Full Pipeline Setup (requires PostgreSQL)

**Prerequisites:** Python 3.11 · PostgreSQL 15+

**1. Configure environment:**
```bash
# .env is auto-created by make setup — edit it to set your database credentials
# AMAZON_DB_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/amazon_sales
# API_KEY=any-random-string-you-choose
```

**2. Run pipeline:**
```bash
make etl       # load data into PostgreSQL (~15 min, one-time)
make eda       # generate 23 EDA charts
make train     # train all 5 models + log to MLflow
make serve &   # start FastAPI on :8000
make app       # start Streamlit dashboard on :8501
```

Dashboard → **http://localhost:8501** · API docs → **http://localhost:8000/docs**

### Option B — Full Docker Stack (advanced, requires Docker Desktop)

```bash
git clone https://github.com/PriyaMonisha/Amazon-India-Sales-Analytics.git
cd Amazon-India-Sales-Analytics
make setup        # create venv, install deps, copy .env
make all          # docker compose up --build (15 services)
```

Full stack → Dashboard **:8501** · API **:8000** · Grafana **:3000** · Airflow **:8080** · MLflow **:5001**

### All Make Commands

| Command | What it does |
|---------|-------------|
| `make setup` | Create venv, install all dependencies, copy `.env.example` → `.env` |
| `make etl` | Run ETL pipeline — extract CSVs, clean, load to PostgreSQL star schema |
| `make eda` | Generate all 23 EDA charts (requires ETL done first) |
| `make train` | Train all 5 ML models and log to MLflow |
| `make serve` | Start FastAPI prediction API on port 8000 |
| `make app` | Start Streamlit dashboard on port 8501 |
| `make feast-apply` | Apply Feast feature store definitions |
| `make feast-materialize` | Materialize features into Redis online store |
| `make monitor` | Start Prometheus + Grafana only (no full stack) |
| `make all` | Full Docker Compose stack — all 15 services |
| `make docs` | Regenerate Analytics Report and Data Dictionary PDFs |
| `make test` | Run full pytest suite (96 tests) |
| `make test-fast` | Fast test run — stop on first failure |
| `make test-cov` | Tests with HTML coverage report |
| `make clean` | Delete all `__pycache__` and `.pyc` files |

---

## Offline Mode

The dashboard works without PostgreSQL — every page shows relevant pre-generated EDA chart images as fallback, with a clear banner explaining how to start the database.

---

## EDA Visualisations (23 Charts)

All charts generated by `notebooks/02_eda.py` and saved to `artifacts/charts/`:

| # | Chart | Analysis Focus |
|---|-------|---------------|
| 01 | Revenue Trend 2015–2025 | Long-term growth trajectory |
| 02 | Seasonal Monthly Heatmap | Seasonality patterns |
| 03 | RFM Customer Segments | Customer segmentation |
| 04 | Payment Method Evolution | Payment behaviour shifts |
| 05 | Subcategory Performance | Category revenue breakdown |
| 06 | Prime Membership Impact | Prime vs non-Prime comparison |
| 07 | Geographic & Tier Analysis | City-tier revenue distribution |
| 08 | Festival Sales Impact | Festival-driven revenue spikes |
| 09 | Customer Tier Behaviour | Tier spending profiles |
| 10 | Age Group Preferences | Demographic purchasing patterns |
| 11 | Price vs Demand | Price elasticity |
| 12 | Brand Performance | Brand market share |
| 13 | Return Rate Analysis | Product return trends |
| 14 | Cohort Retention | Customer retention by cohort |
| 15 | CLV Distribution | Customer lifetime value |
| 16 | Delivery Performance | Delivery speed impact |
| 17 | Discount Effectiveness | Discount ROI analysis |
| 18 | Product Rating by Subcategory | Quality distribution |
| 19 | Pareto Revenue Concentration | 80/20 revenue concentration |
| 20 | YoY Subcategory Growth | Year-on-year growth rates |
| 21 | Customer Journey & Transitions | Category transition mapping |
| 22 | Product Lifecycle Analysis | Launch-to-maturity curves |
| 23 | Competitive Pricing | Brand pricing benchmarks |

---

## API Endpoints

All `/predict/*` and `/monitor/*` endpoints require an `X-API-Key` header:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predict/churn/CUST001
```

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | None | Service health + model load status |
| GET | `/metrics` | None | Prometheus metrics |
| GET | `/predict/churn/{id}` | Required | Churn probability + threshold |
| GET | `/predict/churn/{id}/explain` | Required | SHAP feature contributions |
| GET | `/predict/forecast/{subcategory}` | Required | Demand forecast (12-month) |
| GET | `/predict/pricing/{subcategory}` | Required | Price elasticity + optimal price |
| GET | `/predict/recommend/{subcategory}` | Required | Product recommendations |
| POST | `/predict/anomaly` | Required | Anomaly score for a transaction |
| POST | `/monitor/drift/run` | Required | Trigger Evidently drift report |

---

## Tests

```bash
make test           # full suite (96 tests: 43 ETL · 30 API · 23 model)
make test-fast      # stop on first failure
make test-cov       # with HTML coverage report
```

CI runs automatically on every push via `.github/workflows/ci.yml`.

---

## Production Readiness

| Category | What's Implemented |
|---|---|
| **Security** | API key auth on all endpoints · no secrets in VCS · SHAP explainer in JSON (no pickle) · encoder checksum verification |
| **Performance** | Models preloaded at startup — zero disk I/O per request (<1 ms inference) · vectorised ETL transforms |
| **Reliability** | Model quality gates block serving underperforming models · 96 tests · CI on every push |
| **MLOps** | Versioned model registry · drift-triggered auto-retraining DAG · A/B champion/challenger framework |
| **Observability** | Prometheus request metrics · SHAP feature drift monitoring · Evidently dataset drift · Grafana dashboards |
| **Scalability** | Rate limiting on compute-heavy endpoints · `filelock` atomic writes for concurrent Airflow + API access |

---

## Key Design Decisions

- **Churn label:** 90-day inactivity (not return_status) with right-censoring guard
- **Temporal split:** Train 2015–2023, test 2024 — never random split for time-series
- **RFM scoring:** `rank(pct=True)` + `pd.cut` — avoids "bin edges must be unique" error on skewed data
- **Prophet serialization:** `model_to_json` (not pickle) — portable and version-safe
- **SHAP:** Pre-loaded in FastAPI lifespan once; separate `churn_shap_baseline.json` for drift monitoring
- **Feast:** Two FeatureViews — training inputs vs prediction outputs (never mixed to prevent leakage)
- **Validation:** Pandera over Great Expectations — avoids Windows MAX_PATH issues
- **A/B routing:** SHA-256 hash of `customer_id` → deterministic bucket assignment (same customer always gets same model version)
- **Airflow isolation:** Drift status and registry state written to filesystem — Airflow DAGs never read API in-memory state
- **Model registry:** JSON manifest with `filelock` + `tempfile` atomic writes — safe under concurrent Airflow task execution

---

## Data

| Dataset | Rows | Period |
|---------|------|--------|
| Sales transactions | 1,122,000 | 2015–2025 |
| Product catalog | ~2,000 SKUs | — |

Star schema: `fact_transactions` + `dim_customers` + `dim_products` + `dim_time`

---

## Author

**Priya Monisha** · [GitHub](https://github.com/PriyaMonisha)

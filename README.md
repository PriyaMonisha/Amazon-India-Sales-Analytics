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

| Document | Description |
|----------|-------------|
| [Analytics Report (PDF)](artifacts/reports/Analytics_Report_Amazon_India_Sales_Analytics.pdf) | 23 EDA analyses with business insights, strategic recommendations, and executive summary |
| [Data Dictionary (PDF)](artifacts/reports/Data_Dictionary_Amazon_India_Sales_Analytics.pdf) | Full schema reference — all tables, columns, types, constraints, and business definitions |

> Pre-built PDFs are committed to the repo. Regenerate after code changes with `make docs`.

---

## What This Project Does

| Layer | What's Built |
|-------|-------------|
| **Data Engineering** | ETL pipeline processing 11 yearly CSVs into a PostgreSQL star schema (1.1M rows, 4 tables, 10 cleaning challenges) |
| **Feature Store** | Feast + Redis for online/offline feature serving; 14 customer features; RFM segmentation |
| **ML Models** | 5 models: churn prediction, demand forecasting, price elasticity, product recommendations, anomaly detection |
| **Model Registry** | Versioned artifact registry with atomic writes, promotion log, and one-command rollback |
| **A/B Testing** | Hash-based deterministic champion/challenger routing with nightly outcome materialization and auto-promotion DAG |
| **API** | FastAPI with 5 prediction endpoints, API key auth, rate limiting, and `/health/models` registry endpoint |
| **Monitoring** | Evidently drift detection · Prometheus metrics · SHAP feature importance drift · 4 Grafana dashboards |
| **Auto-Retraining** | Airflow DAG triggered by persisted drift results — retrains only when drift exceeds configurable threshold |
| **Dashboard** | Streamlit app with **10 pages and 50+ interactive charts** |
| **Orchestration** | 7 Airflow DAGs (ETL, inference, model retraining, A/B promotion) |
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
          ┌──────────┴──────────────┐
          ▼                         ▼
  Streamlit Dashboard       Prometheus + Grafana
  (10 pages, 50+ charts)   (4 dashboards, drift alerts)

  Airflow Orchestration (7 DAGs)
  ├── ETL + inference pipeline
  ├── Drift-triggered model retraining
  └── A/B champion/challenger promotion
```

---

## Dashboard Pages

| Page | Analytics Coverage | Key Charts |
|------|--------------------|------------|
| **0 — Executive Dashboard** | KPI monitoring, strategic overview, financial performance | KPI alerts, YoY growth, market share, BI command centre |
| **1 — Customer Churn** | Churn prediction, customer segmentation | XGBoost probability distribution, SHAP waterfall, live predictor |
| **2 — Demand Forecast** | Sales forecasting, seasonal trends | Prophet forecast + CI bands, YoY growth, seasonal heatmap |
| **3 — Pricing Analytics** | Price optimisation, demand sensitivity | Revenue scenarios, demand sensitivity, price-revenue bubble chart |
| **4 — Recommendations** | Product recommendations, basket analysis | Association rules, confidence-lift scatter, co-purchase heatmap |
| **5 — Anomaly Detection** | Revenue + return anomalies | Score gauge, distribution comparisons, live transaction checker |
| **6 — Festival & Seasonal** | Festival impact, seasonality patterns | Festival revenue comparison, seasonal pattern, subcategory breakdown |
| **7 — Prime & Demographics** | Prime member analysis, demographics | Prime KPI cards, category preferences, age group spending |
| **8 — Brand & Products** | Brand performance, product lifecycle | Brand market share, product ratings, new launch performance |
| **9 — Customer Journey** | Purchase frequency, CLV, journey mapping | Purchase frequency, category transitions, CLV by segment |

---

## ML Models

### 1. Customer Churn Prediction
- **Algorithm:** XGBoost with Optuna hyperparameter tuning (50 trials, stratified CV)
- **Label:** No purchase in next 90 days — temporal definition, not return status
- **Split:** Temporal — train on 2015–2023, test on 2024 cohort (no data leakage)
- **Features:** 14 RFM + behavioural features from Feast feature store
- **Explainability:** SHAP values per prediction via `/predict/churn/{id}/explain`

### 2. Demand Forecasting
- **Algorithm:** Facebook Prophet per product subcategory
- **Seasonality:** Multiplicative; custom Diwali/festival holiday windows
- **Granularity:** Monthly per subcategory; 12-month forward horizon
- **Serialization:** `model_to_json` — portable across environments, no pickle

### 3. Price Elasticity
- **Algorithm:** XGBoost log-log regression (price → demand)
- **Output:** Elasticity coefficient + revenue-maximising price at ±10%, ±20% scenarios

### 4. Product Recommendations
- **Algorithm:** FP-Growth on order-level baskets (mlxtend)
- **Fallback:** Progressive min_support search → global popularity ranking (cold-start safe)

### 5. Anomaly Detection
- **Algorithm:** IsolationForest on transaction features
- **Output:** Continuous anomaly score + binary flag per transaction

---

## Tech Stack

```
Python 3.11        XGBoost 2.0        Prophet 1.1.5
scikit-learn 1.5   mlxtend 0.23       Optuna 3.x
MLflow 2.14        Feast 0.38         Redis 5.0
FastAPI 0.115      Streamlit 1.35     Evidently 0.4
Airflow 2.8        PostgreSQL 15      Prometheus + Grafana
Docker Compose     Pandera            SHAP 0.44
```

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_engineering.py    # ETL → PostgreSQL star schema
│   ├── 02_eda.py                 # 23 EDA charts saved to artifacts/charts/
│   └── 03_ml_training.py         # All 5 models + MLflow logging
├── src/
│   ├── etl/                      # extract.py, transform.py, load.py
│   ├── features/                 # compute.py (RFM), feature_store.py (Feast + Redis)
│   ├── models/                   # churn, forecasting, pricing, recommendation, anomaly
│   ├── model_registry/           # Versioned artifact registry with atomic writes + rollback
│   ├── monitoring/               # Prometheus metrics, Evidently drift, SHAP drift monitoring
│   ├── ab_testing/               # Hash-based routing, AUC comparison, outcome tracking
│   ├── utils/                    # Shared MLflow setup
│   └── validation/               # Pandera schema validation (7 business rules)
├── api/
│   ├── main.py                   # FastAPI lifespan model loading + A/B challenger
│   ├── dependencies.py           # API key auth + rate limiter
│   ├── models.py                 # Pydantic request/response schemas
│   └── routers/                  # churn, forecast, pricing, recommendation, anomaly
├── streamlit_app/
│   ├── app.py                    # Overview page (KPIs, trends, geography)
│   ├── pages/                    # 9 additional pages
│   └── utils/                    # db.py, api_client.py, offline.py
├── dags/                         # 7 Airflow DAGs
├── scripts/                      # CLI wrappers for Airflow-isolated training
├── feast_repo/                   # Feature views, entities, feature services
├── monitoring/                   # Grafana dashboards, Prometheus config
├── tests/                        # 96 tests: test_etl.py, test_api.py, test_models.py
├── requirements/                 # base.txt, dev.txt, api.txt, ml.txt, etl.txt, ci.txt
├── .github/workflows/ci.yml      # Lint + 96 tests on every push
├── Dockerfile.api
├── Dockerfile.streamlit
└── docker-compose.yml            # 15-service full stack
```

---

## Quick Start

### Verify the code — no database or data files needed

```bash
git clone https://github.com/PriyaMonisha/Amazon-India-Sales-Analytics.git
cd Amazon-India-Sales-Analytics
make setup    # creates venv, installs dependencies, copies .env.example → .env
make test     # 96 tests pass — uses mocks, no database or data files required
```

Tests cover ETL transformations, all 5 API prediction endpoints, and model training logic using in-memory fixtures. Raw CSVs and trained model artifacts are not committed to the repo (size); the test suite verifies correctness without them.

---

### Full pipeline (requires Python 3.11 + PostgreSQL 15+)

**Step 1 — Set your database credentials in `.env`** (auto-created by `make setup`):

```
AMAZON_DB_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/amazon_sales
API_KEY=any-string-you-choose
```

**Step 2 — Run the pipeline:**

```bash
make etl       # load data into PostgreSQL (~15 min, one-time)
make eda       # generate 23 EDA charts
make train     # train all 5 models + log to MLflow
make serve &   # start FastAPI on :8000
make app       # start Streamlit dashboard on :8501
```

Dashboard → **http://localhost:8501** · API docs → **http://localhost:8000/docs**

---

### Full Docker stack (requires Docker Desktop)

```bash
make setup    # creates venv + copies .env
make all      # docker compose up --build (15 services)
```

Full stack → Dashboard **:8501** · API **:8000** · Grafana **:3000** · Airflow **:8080** · MLflow **:5001**

---

### Make commands reference

| Command | What it does |
|---------|-------------|
| `make setup` | Create venv, install dependencies, copy `.env.example` → `.env` |
| `make etl` | Extract CSVs, clean data, load to PostgreSQL star schema |
| `make eda` | Generate all 23 EDA charts |
| `make train` | Train all 5 ML models and log to MLflow |
| `make serve` | Start FastAPI prediction API on port 8000 |
| `make app` | Start Streamlit dashboard on port 8501 |
| `make feast-apply` | Apply Feast feature store definitions |
| `make feast-materialize` | Materialize features into Redis online store |
| `make monitor` | Start Prometheus + Grafana only |
| `make all` | Full Docker Compose stack — all 15 services |
| `make docs` | Regenerate Analytics Report and Data Dictionary PDFs |
| `make test` | Full pytest suite (96 tests) |
| `make test-cov` | Tests with HTML coverage report |
| `make clean` | Delete all `__pycache__` and `.pyc` files |

---

## API Endpoints

All `/predict/*` and `/monitor/*` endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predict/churn/CUST001
```

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | None | Service health + model load status |
| GET | `/health/models` | None | Registry version + A/B challenger status |
| GET | `/metrics` | None | Prometheus metrics |
| GET | `/predict/churn/{id}` | Required | Churn probability + threshold |
| GET | `/predict/churn/{id}/explain` | Required | SHAP feature contributions (top 10) |
| GET | `/predict/forecast/{subcategory}` | Required | 12-month demand forecast |
| GET | `/predict/pricing/{subcategory}` | Required | Price elasticity + optimal price scenarios |
| GET | `/predict/recommend/{subcategory}` | Required | Product recommendations |
| POST | `/predict/anomaly` | Required | Anomaly score for a transaction |
| POST | `/monitor/drift/run` | Required | Trigger Evidently drift report |

---

## Tests

```bash
make test        # 96 tests: 43 ETL · 30 API · 23 model
make test-cov    # with HTML coverage report
```

CI runs on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## Production Readiness

| Category | What's Implemented |
|---|---|
| **Security** | API key auth · no secrets in VCS · SHAP explainer in JSON (no pickle) · encoder SHA-256 checksum |
| **Performance** | Models preloaded at startup — zero disk I/O per request (<1 ms inference) · vectorised ETL |
| **Reliability** | Model quality gates block underperforming models · 96 tests · CI on every push |
| **MLOps** | Versioned model registry · drift-triggered auto-retraining · A/B champion/challenger framework |
| **Observability** | Prometheus request metrics · SHAP feature drift · Evidently dataset drift · Grafana dashboards |
| **Scalability** | Rate limiting on compute-heavy endpoints · `filelock` atomic writes for concurrent access |

---

## Key Design Decisions

- **Churn label:** 90-day inactivity window with right-censoring guard — not return status
- **Temporal split:** Train 2015–2023, test 2024 cohort — no random splits for time-series data
- **RFM scoring:** `rank(pct=True)` + `pd.cut` — handles skewed distributions without bin-edge errors
- **Prophet serialization:** `model_to_json` — portable across environments, avoids pickle fragility
- **Feature store:** Two separate Feast FeatureViews — training inputs and prediction outputs are never mixed (prevents leakage)
- **A/B routing:** SHA-256 hash of `customer_id` — deterministic, same customer always gets same model version
- **Airflow isolation:** Drift status written to `latest_drift.json` — DAGs read from disk, never from API memory
- **Model registry:** JSON manifest with `filelock` + atomic `tempfile` writes — safe under concurrent Airflow execution

---

## Data

| Dataset | Rows | Period |
|---------|------|--------|
| Sales transactions | 1,122,000 | 2015–2025 |
| Product catalog | ~2,000 SKUs | — |

Star schema: `fact_transactions` · `dim_customers` · `dim_products` · `dim_time`

---

## Offline Mode

The Streamlit dashboard works without a running database — every page falls back to pre-generated EDA chart images with a clear banner explaining how to start the full stack.

---

## Author

**Priya Monisha** · [GitHub](https://github.com/PriyaMonisha)

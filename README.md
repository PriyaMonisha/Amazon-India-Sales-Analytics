# Amazon India Sales Analytics — End-to-End ML Platform

> **11 years of Amazon India sales data (2015–2025) · 1.1 million transactions · 5 production ML models · Full MLOps stack**

A production-grade machine learning platform built on a decade of Amazon India e-commerce data. Covers the complete ML lifecycle: raw data ingestion → feature engineering → model training → real-time serving → monitoring → interactive dashboard.

---

## What This Project Does

| Layer | What's Built |
|-------|-------------|
| **Data Engineering** | ETL pipeline processing 11 yearly CSVs into a PostgreSQL star schema (1.1M rows, 4 tables) |
| **Feature Store** | Feast + Redis for online/offline feature serving; 14 customer features; RFM segmentation |
| **ML Models** | 5 models: churn prediction, demand forecasting, price elasticity, product recommendations, anomaly detection |
| **API** | FastAPI with 5 prediction endpoints + `/health` + Prometheus `/metrics` |
| **Monitoring** | Evidently data drift detection, Prometheus metrics, 4 Grafana dashboards |
| **Dashboard** | Streamlit app with 6 pages and 30 interactive charts |
| **Orchestration** | 5 Airflow DAGs chained via TriggerDagRunOperator |
| **Infra** | Full Docker Compose stack (15 services) with health checks and volume mounts |
| **Tests** | 95 pytest tests across ETL, API, and model layers |

---

## Architecture

```
Raw CSVs (11 years)
      │
      ▼
 ETL Pipeline ──────────────────────► PostgreSQL (star schema)
 (extract / transform / validate)          │
                                           │
                              ┌────────────┴───────────┐
                              ▼                        ▼
                      Feature Store             EDA (21 charts)
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
     (6 pages, 30 charts)  (4 dashboards, drift alerts)
                    │
                    ▼
            Airflow Orchestration
          (5 DAGs, weekly ETL → daily drift)
```

---

## ML Models

### 1. Customer Churn Prediction
- **Algorithm:** XGBoost with Optuna hyperparameter tuning
- **Label:** No purchase in next 90 days (temporal definition, not return status)
- **Split:** Temporal — train on 2015–2023, test on 2024 cohort
- **Features:** 14 RFM + behavioral features from Feast feature store
- **Explainability:** SHAP values served per prediction via FastAPI
- **Artifacts:** model, SHAP explainer, encoder, feature names, threshold

### 2. Demand Forecasting (Prophet)
- **Algorithm:** Facebook Prophet per product subcategory
- **Seasonality:** Multiplicative; custom Diwali/festival regressors
- **Granularity:** Monthly, per subcategory; 12-month horizon
- **Serialization:** `model_to_json` (not pickle — portable across environments)
- **Metric:** WMAPE (weighted mean absolute percentage error)

### 3. Price Elasticity
- **Algorithm:** XGBoost log-log regression (price → demand)
- **Output:** Elasticity coefficient + revenue-maximizing price per SKU
- **Bucketing:** Quantile-based with rank fallback for sparse distributions

### 4. Product Recommendations (FP-Growth)
- **Algorithm:** mlxtend FP-Growth on order-level baskets
- **Fallback:** Progressive min_support [0.005 → 0.002 → 0.001] + popularity fallback
- **Rules:** Filtered by confidence + lift thresholds

### 5. Anomaly Detection
- **Algorithm:** IsolationForest on transaction features
- **Features:** Discount %, delivery days, price tier, payment method
- **Output:** Anomaly score + binary flag via FastAPI

---

## Tech Stack

```
Python 3.11        XGBoost 2.0.3      Prophet 1.1.5
scikit-learn 1.5   mlxtend 0.23       Optuna 3.x
MLflow 2.14        Feast 0.38         Redis 5.0
FastAPI 0.115      Streamlit 1.37     Evidently 0.4.30
Airflow 2.8        PostgreSQL 15      Prometheus + Grafana
Docker Compose     Pandera            SHAP
```

---

## Project Structure

```
├── src/
│   ├── etl/              # extract.py, transform.py, load.py
│   ├── features/         # compute.py (RFM, customer features), feature_store.py
│   ├── models/           # churn.py, forecasting.py, pricing.py, recommendation.py, anomaly.py
│   ├── monitoring/       # metrics.py (Prometheus), drift.py (Evidently)
│   └── validation/       # expectations.py (Pandera, 7 schema checks)
├── api/
│   ├── main.py           # FastAPI app with lifespan model loading
│   ├── models.py         # Pydantic request/response schemas
│   └── routers/          # churn, forecast, pricing, recommendation, anomaly
├── streamlit_app/
│   ├── app.py            # Overview page (KPIs, trends, geography)
│   └── pages/            # 5 feature pages (churn, forecast, pricing, reco, anomaly)
├── dags/                 # 5 Airflow DAGs (ETL → features → training → serving → drift)
├── feast_repo/           # Feature views, entities, feature services
├── monitoring/           # Grafana dashboards, Prometheus config
├── notebooks/            # 01_data_cleaning.py, 01_data_engineering.py, 02_eda.py, 03_ml_training.py
├── tests/                # test_etl.py (43), test_api.py (30), test_models.py (22)
├── requirements/         # base.txt, dev.txt, airflow.txt
├── Dockerfile.api
├── Dockerfile.streamlit
└── docker-compose.yml    # 15-service stack
```

---

## Quick Start

### Prerequisites
- Python 3.11
- Docker Desktop (for PostgreSQL, Redis, full stack)
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/PriyaMonisha/Amazon-India-Sales-Analytics.git
cd Amazon-India-Sales-Analytics

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements/base.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD, REDIS_URL, MLFLOW_TRACKING_URI
```

### 3. Start Infrastructure

```bash
docker compose up postgres redis -d
```

### 4. Run ETL Pipeline

```bash
python notebooks/01_data_engineering.py
```

### 5. Compute Features & Apply Feast

```bash
python -c "from src.features.compute import compute_customer_features; compute_customer_features()"
cd feast_repo && feast apply && cd ..
python -c "from src.features.feature_store import materialize_features; materialize_features()"
```

### 6. Train Models

```bash
# Requires Docker (PostgreSQL must be running)
python notebooks/03_ml_training.py
```

### 7. Start FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 8. Launch Dashboard

```bash
streamlit run streamlit_app/app.py
```

### Full Stack (Docker Compose)

```bash
docker compose up
```

Services: PostgreSQL · Redis · FastAPI (port 8000) · Streamlit (port 8501) · Airflow (port 8080) · Prometheus (port 9090) · Grafana (port 3000)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health + model load status |
| GET | `/metrics` | Prometheus metrics |
| POST | `/predict/churn` | Churn probability + SHAP explanation |
| POST | `/predict/forecast` | Demand forecast (12-month) |
| POST | `/predict/pricing` | Price elasticity + optimal price |
| POST | `/predict/recommendations` | Product recommendations |
| POST | `/predict/anomaly` | Anomaly score for a transaction |
| POST | `/monitor/drift/run` | Trigger Evidently drift report |

---

## Dashboard Pages

| Page | Key Visuals |
|------|-------------|
| **Overview** | Revenue trend (11yr), top subcategories, state heatmap, payment method evolution |
| **Customer Churn** | Score distribution, confusion matrix, SHAP waterfall, live predictor |
| **Demand Forecast** | Prophet forecast + CI bands, YoY growth, seasonal heatmap |
| **Pricing Analytics** | Revenue scenarios, demand sensitivity, price-revenue bubble chart |
| **Recommendations** | Association rules, confidence-lift scatter, co-purchase heatmap |
| **Anomaly Detection** | Score gauge, distribution comparisons, live transaction checker |

---

## Tests

```bash
pip install -r requirements/dev.txt
pytest tests/ -v
# 95 tests: 43 ETL · 30 API · 22 model
```

---

## Key Design Decisions

- **Churn label:** 90-day inactivity (not return_status) with right-censoring guard
- **Temporal split:** Train 2015–2023, test 2024 cohort — never random split for time-series
- **RFM scoring:** `rank(pct=True)` + `pd.cut` — avoids "bin edges must be unique" error on skewed e-commerce data
- **Prophet serialization:** `model_to_json` (not pickle) — portable and version-safe
- **SHAP:** Pre-loaded in FastAPI lifespan once; not per-request
- **Docker memory:** `--workers 1` for uvicorn — 5 Prophet models × N workers = OOM
- **Feast:** Two FeatureViews — training inputs vs prediction outputs (never mixed to prevent leakage)
- **Validation:** Pandera over Great Expectations — avoids Windows MAX_PATH issues with ipywidgets

---

## Data

| Dataset | Rows | Period |
|---------|------|--------|
| Sales transactions | 1,122,000 | 2015–2025 |
| Product catalog | ~50,000 SKUs | — |

Star schema: `fact_transactions` + `dim_customers` + `dim_products` + `dim_time`

---

## Author

**Priya Monisha** · [GitHub](https://github.com/PriyaMonisha)

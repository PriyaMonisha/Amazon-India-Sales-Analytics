from enum import Enum
from pathlib import Path
import os


class Env(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    AIRFLOW = "airflow"


APP_ENV = Env(os.getenv("APP_ENV", "local"))
RANDOM_STATE = 42
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"

# --- Database (postgresql+psycopg2:// required by SQLAlchemy) ---
# LOCAL uses port 5433 because 5432 is taken by a pre-existing local PostgreSQL.
# Docker containers talk to each other on port 5432 (internal Docker network).
DB_URLS = {
    Env.LOCAL:   os.getenv(
        "AMAZON_DB_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/amazon_sales",
    ),
    Env.DOCKER:  "postgresql+psycopg2://postgres:postgres@postgres:5432/amazon_sales",
    Env.AIRFLOW: os.getenv(
        "AMAZON_DB_URL",  # custom var — NOT AIRFLOW_CONN_POSTGRES_DEFAULT
        "postgresql+psycopg2://postgres:postgres@postgres:5432/amazon_sales"
    ),
}
DB_URL: str = DB_URLS[APP_ENV]

# --- Redis ---
REDIS_HOSTS = {Env.LOCAL: "localhost", Env.DOCKER: "redis", Env.AIRFLOW: "redis"}
REDIS_HOST: str = REDIS_HOSTS[APP_ENV]
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = 0

# --- MinIO (local S3-compatible, replaces GCS) ---
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET: str = "amazon-sales"
MINIO_SECURE: bool = False

# --- MLflow ---
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT: str = "amazon_sales_models"

# --- Paths ---
BASE_DIR: Path = Path(__file__).resolve().parent
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
FEAST_REPO_PATH: Path = BASE_DIR / "feast_repo"

# --- FastAPI ---
FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
FASTAPI_URL: str = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

# --- Model thresholds ---
CHURN_THRESHOLD: float = float(os.getenv("CHURN_THRESHOLD", "0.5"))
DRIFT_THRESHOLD: float = 0.10
CHURN_MIN_ROC_AUC: float = 0.72
FORECAST_MAX_WMAPE: float = 0.25
PRICING_MIN_R2: float = 0.45
DRIFT_RETRAIN_THRESHOLD: float = float(os.getenv("DRIFT_RETRAIN_THRESHOLD", "0.30"))

# --- FAST_MODE params ---
N_OPTUNA_TRIALS: int = 10 if FAST_MODE else 50
CV_FOLDS: int = 3 if FAST_MODE else 5
SAMPLE_ROWS: int | None = 50_000 if FAST_MODE else None  # None = full dataset

# --- ETL ---
ETL_CHUNK_SIZE: int = 10_000

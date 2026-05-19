# %% [markdown]
# # Section 1: Data Engineering — ETL Pipeline
#
# Flow: Raw CSVs → Extract → Clean (10 challenges) → GE Validate → Load to PostgreSQL star schema
#
# Run locally: `make etl`  (APP_ENV=local python notebooks/01_data_engineering.py)
# Run in Docker: uses APP_ENV=docker (postgres service name resolves)

# %%
import logging
import sys
from pathlib import Path

# PROJECT_ROOT: works as .py script (terminal) and Jupyter/Colab
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etl_notebook")

# %%
import config
from src.etl.extract import load_raw_csvs, load_product_catalog
from src.etl.transform import clean_sales, clean_products, build_cleaning_log
from src.etl.load import create_tables, get_engine, load_dimensions, load_facts

logger.info(f"APP_ENV={config.APP_ENV} | FAST_MODE={config.FAST_MODE}")
logger.info(f"DB_URL: {config.DB_URL[:60]}...")
logger.info(f"RAW_DATA_DIR: {config.RAW_DATA_DIR}")

# %% [markdown]
# ## Step 1: Extract raw CSVs

# %%
# Reference project data directory (11 yearly CSVs)
# Adjust this path to point to where the CSVs live on your machine.
# Default: data/raw/ within this project, or the reference project folder.
REF_DATA_DIR = Path(r"c:\Users\Suba\Documents\Data science\Guvi\Proj Amazon_India_Sales_Analytics - 2\amazon_sales_datasets")
REF_PRODUCT_DIR = Path(r"c:\Users\Suba\Documents\Data science\Guvi\Proj Amazon_India_Sales_Analytics - 2\product_dataset")

# Use reference data dir (no need to copy 300MB of CSVs)
RAW_SALES_DIR = REF_DATA_DIR if REF_DATA_DIR.exists() else config.RAW_DATA_DIR

logger.info(f"Loading CSVs from: {RAW_SALES_DIR}")
df_raw = load_raw_csvs(RAW_SALES_DIR)
logger.info(f"Raw sales: {len(df_raw):,} rows × {len(df_raw.columns)} columns")

# %%
# Load product catalog
PRODUCT_DIR = REF_PRODUCT_DIR if REF_PRODUCT_DIR.exists() else config.RAW_DATA_DIR.parent / "product_dataset"
df_products_raw = load_product_catalog(PRODUCT_DIR)
logger.info(f"Product catalog: {len(df_products_raw):,} rows")

# %%
# Preview
print(df_raw.shape)
print(df_raw.dtypes.value_counts())
print(df_raw.head(3).to_string())

# %% [markdown]
# ## Step 2: Transform (all 10 cleaning challenges)

# %%
if config.FAST_MODE and config.SAMPLE_ROWS:
    logger.info(f"FAST_MODE: sampling {config.SAMPLE_ROWS:,} rows for development")
    df_sample = df_raw.sample(n=config.SAMPLE_ROWS, random_state=config.RANDOM_STATE)
else:
    df_sample = df_raw
    logger.info("FAST_MODE=False: transforming full dataset")

df_clean, cleaning_log = clean_sales(df_sample)
df_products_clean = clean_products(df_products_raw)

logger.info(f"After cleaning: {len(df_clean):,} rows × {len(df_clean.columns)} columns")
logger.info(f"Cleaning log: {cleaning_log}")

# %%
# Save cleaning log (with timestamp to avoid same-day overwrites)
build_cleaning_log(cleaning_log, config.ARTIFACTS_DIR)

# %%
# Preview cleaned data
print("\n--- Cleaned dtypes ---")
print(df_clean.dtypes.to_string())
print("\n--- Null counts (top 10) ---")
print(df_clean.isnull().sum().sort_values(ascending=False).head(10))
print("\n--- Unique counts (selected cols) ---")
for col in ["payment_method", "return_status", "is_festival_sale", "customer_tier"]:
    if col in df_clean.columns:
        print(f"  {col}: {df_clean[col].nunique()} unique | sample: {df_clean[col].value_counts().head(3).to_dict()}")

# %% [markdown]
# ## Step 3: Validate with Great Expectations (v0.18.19)

# %%
from src.validation.expectations import validate_sales, validate_products

try:
    validate_sales(df_clean)
    validate_products(df_products_clean)
    logger.info("GE validation PASSED for both datasets.")
except RuntimeError as e:
    logger.error(f"GE validation FAILED: {e}")
    # In Airflow: this would mark the task failed and stop downstream DAGs.
    # In notebook mode: print the error and continue for debugging.
    print(f"WARNING: {e}")

# %% [markdown]
# ## Step 4: Load to PostgreSQL star schema

# %%
engine = get_engine(config.DB_URL)

logger.info("Creating / verifying star schema tables...")
create_tables(engine)

logger.info("Loading dimension tables...")
load_dimensions(df_clean, df_products_clean, engine)

logger.info("Loading fact_transactions (upsert — safe to re-run)...")
n_rows = load_facts(df_clean, engine, chunk_size=config.ETL_CHUNK_SIZE)

# %%
# Verify counts in PostgreSQL
from sqlalchemy import text

with engine.connect() as conn:
    for table in ["fact_transactions", "dim_customers", "dim_products", "dim_time"]:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"  {table}: {count:,} rows")

logger.info(f"ETL complete. {n_rows:,} rows loaded into fact_transactions.")
print("\nSection 1 complete. Run `make eda` next.")

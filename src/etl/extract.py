import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_csvs(data_dir: Path, file_pattern: str = "amazon_india_20*.csv") -> pd.DataFrame:
    """
    Load all yearly Amazon India CSVs from data_dir, standardize column names, and concat.

    Returns a combined DataFrame with a 'source_year' column derived from filename.
    Skips missing or empty files with a warning.
    """
    file_paths = sorted(data_dir.glob(file_pattern))

    if not file_paths:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in {data_dir}. "
            f"Copy the 11 yearly CSVs into data/raw/ first."
        )

    logger.info(f"Found {len(file_paths)} files in {data_dir}")
    dfs: list[pd.DataFrame] = []

    for file_path in file_paths:
        logger.info(f"Loading: {file_path.name}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logger.warning(f"Skipping {file_path.name}: {e}")
            continue

        if df.empty:
            logger.warning(f"Skipping empty file: {file_path.name}")
            continue

        # Standardize column names: strip whitespace, lowercase, spaces → underscores
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )

        # Extract year from filename (e.g. amazon_india_2019.csv → 2019)
        stem = file_path.stem  # e.g. "amazon_india_2019"
        parts = stem.split("_")
        year_str = parts[-1] if parts[-1].isdigit() else ""
        if year_str:
            df["source_year"] = int(year_str)
        else:
            logger.warning(f"Could not parse year from filename: {file_path.name}")

        dfs.append(df)
        logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")

    if not dfs:
        raise ValueError("No valid data loaded. Check data/raw/ directory.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} rows from {len(dfs)} files")
    return combined


def load_product_catalog(data_dir: Path, filename: str = "amazon_india_products_catalog.csv") -> pd.DataFrame:
    """Load the product catalog CSV from data_dir."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Product catalog not found at: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    logger.info(f"Product catalog: {len(df):,} rows, {len(df.columns)} columns")
    return df

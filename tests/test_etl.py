import pandas as pd
import pytest

from conftest import _make_minimal_valid_sales_df


# ---------------------------------------------------------------------------
# TestExtract (6 tests)
# ---------------------------------------------------------------------------

@pytest.mark.etl
class TestExtract:
    def test_year_extraction(self, tmp_path):
        from src.etl.extract import load_raw_csvs
        for year in [2022, 2023]:
            p = tmp_path / f"amazon_india_{year}.csv"
            p.write_text("transaction_id,final_amount_inr\nT001,1000\n")
        df = load_raw_csvs(tmp_path)
        assert set(df["source_year"].unique()) == {2022, 2023}

    def test_column_standardization(self, tmp_path):
        from src.etl.extract import load_raw_csvs
        p = tmp_path / "amazon_india_2022.csv"
        p.write_text("Transaction ID,Final Amount INR\nT001,1000\n")
        df = load_raw_csvs(tmp_path)
        assert "transaction_id" in df.columns

    def test_raises_file_not_found(self, tmp_path):
        from src.etl.extract import load_raw_csvs
        with pytest.raises(FileNotFoundError):
            load_raw_csvs(tmp_path)

    def test_raises_value_error_empty_csv(self, tmp_path):
        from src.etl.extract import load_raw_csvs
        p = tmp_path / "amazon_india_2022.csv"
        p.write_text("transaction_id,final_amount_inr\n")  # header only
        with pytest.raises(ValueError):
            load_raw_csvs(tmp_path)

    def test_catalog_success(self, tmp_path):
        from src.etl.extract import load_product_catalog
        p = tmp_path / "amazon_india_products_catalog.csv"
        p.write_text("product_id,product_name\nP001,Widget\n")
        df = load_product_catalog(tmp_path)
        assert len(df) == 1
        assert "product_id" in df.columns

    def test_catalog_raises_file_not_found(self, tmp_path):
        from src.etl.extract import load_product_catalog
        with pytest.raises(FileNotFoundError):
            load_product_catalog(tmp_path)


# ---------------------------------------------------------------------------
# TestTransformParsers (13 tests — parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.etl
class TestTransformParsers:
    @pytest.mark.parametrize("raw,expected", [
        ("₹1,299", 1299.0),
        ("₹ 1,29,999", 129999.0),
        (None, None),
        ("price on request", None),
        ("-", None),
        ("999.99", 999.99),
    ])
    def test_parse_price(self, raw, expected):
        from src.etl.transform import _parse_price
        result = _parse_price(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize("raw,expected", [
        ("4 stars", 4.0),
        ("4 star", 4.0),   # regex r"^\d+\s*stars?$" handles singular — verified transform.py:110
        ("3/5", 3.0),
        ("2.5/5.0", 2.5),
        ("4.5", 4.5),
        (None, None),
        ("nan", None),
    ])
    def test_parse_rating(self, raw, expected):
        from src.etl.transform import _parse_rating
        result = _parse_rating(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestTransformCleanSales (10 tests)
# ---------------------------------------------------------------------------

@pytest.mark.etl
class TestTransformCleanSales:
    def test_clean_sales_returns_tuple(self, sample_sales_df):
        from src.etl.transform import clean_sales
        result = clean_sales(sample_sales_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_clean_sales_returns_dataframe(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        assert isinstance(cleaned, pd.DataFrame)

    def test_clean_sales_log_is_dict(self, sample_sales_df):
        from src.etl.transform import clean_sales
        _, log = clean_sales(sample_sales_df)
        assert isinstance(log, dict)

    def test_clean_sales_preserves_row_count(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        assert len(cleaned) == len(sample_sales_df)

    def test_clean_sales_has_required_columns(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        for col in ["transaction_id", "customer_id", "final_amount_inr"]:
            assert col in cleaned.columns

    def test_clean_sales_order_date_is_datetime(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        if "order_date" in cleaned.columns:
            assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])

    def test_clean_sales_no_negative_amounts(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        if "final_amount_inr" in cleaned.columns:
            valid = cleaned["final_amount_inr"].dropna()
            assert (valid >= 0).all()

    def test_clean_sales_rating_in_range(self, sample_sales_df):
        from src.etl.transform import clean_sales
        cleaned, _ = clean_sales(sample_sales_df)
        if "customer_rating" in cleaned.columns:
            valid = cleaned["customer_rating"].dropna()
            assert ((valid >= 0) & (valid <= 5)).all()

    def test_clean_sales_does_not_modify_input(self, sample_sales_df):
        from src.etl.transform import clean_sales
        original_len = len(sample_sales_df)
        clean_sales(sample_sales_df)
        assert len(sample_sales_df) == original_len

    def test_clean_products_returns_dataframe(self):
        from src.etl.transform import clean_products
        result = clean_products(pd.DataFrame({"product_id": ["P1"]}))
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestLoad (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.etl
class TestLoad:
    def test_upsert_empty_returns_zero(self, mock_engine):
        from src.etl.load import _upsert_dataframe
        result = _upsert_dataframe(pd.DataFrame(), "fact_transactions", "transaction_id", mock_engine)
        assert result == 0

    def test_upsert_returns_real_count(self, mock_engine):
        from src.etl.load import _upsert_dataframe
        df = _make_minimal_valid_sales_df(n=10)
        result = _upsert_dataframe(df, "fact_transactions", "transaction_id", mock_engine)
        assert result == 10  # from len(records), not mock return value

    def test_get_engine_creates_engine(self):
        from src.etl.load import get_engine
        from sqlalchemy import Engine
        engine = get_engine("sqlite:///:memory:")
        assert isinstance(engine, Engine)

    def test_load_dimensions_no_exception(self, mock_engine, sample_sales_df, sample_products_df):
        from src.etl.load import load_dimensions
        load_dimensions(sample_sales_df, sample_products_df, mock_engine)
        assert mock_engine.begin.called, "load_dimensions() never called engine.begin() — no DB writes attempted"


# ---------------------------------------------------------------------------
# TestValidation (10 tests)
# ---------------------------------------------------------------------------

@pytest.mark.etl
class TestValidation:
    def test_passes_fast_mode(self, sample_sales_df):
        from src.validation.expectations import validate_sales
        assert validate_sales(sample_sales_df, fast_mode=True) is True

    def test_raises_out_of_range_rating(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=5)
        df.loc[0, "customer_rating"] = 6.0  # > 5.0 — invalid
        with pytest.raises(RuntimeError):
            validate_sales(df, fast_mode=True)

    def test_raises_out_of_range_delivery(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=5)
        df.loc[0, "delivery_days"] = 31.0  # > 30 — invalid
        with pytest.raises(RuntimeError):
            validate_sales(df, fast_mode=True)

    def test_raises_null_customer_id(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=5)
        df.loc[0, "customer_id"] = None
        with pytest.raises(RuntimeError):
            validate_sales(df, fast_mode=True)

    def test_raises_duplicate_transaction_id(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=5)
        df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]  # duplicate
        with pytest.raises(RuntimeError, match="transaction_id"):
            validate_sales(df, fast_mode=True)

    def test_raises_negative_amount(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=5)
        df.loc[0, "final_amount_inr"] = -1.0
        with pytest.raises(RuntimeError):
            validate_sales(df, fast_mode=True)

    def test_fast_mode_true_skips_row_count(self):
        from src.validation.expectations import validate_sales
        df = _make_minimal_valid_sales_df(n=10)
        assert validate_sales(df, fast_mode=True) is True

    def test_fast_mode_false_enforces_row_count(self):
        from src.validation.expectations import validate_sales
        # Error includes "Row count outside expected range [1,000,000, 1,300,000]"
        df = _make_minimal_valid_sales_df(n=10)
        with pytest.raises(RuntimeError, match="Row count"):
            validate_sales(df, fast_mode=False)

    def test_validate_products_passes_large_df(self, sample_products_df):
        from src.validation.expectations import validate_products
        large_df = pd.concat([sample_products_df] * 35, ignore_index=True)
        large_df["product_id"] = [f"PROD{i:05d}" for i in range(len(large_df))]
        assert validate_products(large_df) is True

    def test_validate_products_raises_small_df(self, sample_products_df):
        from src.validation.expectations import validate_products
        # 50 rows < MIN_ROWS_PRODUCTS=1500
        # Error: RuntimeError("Product schema failed: ... Product row count outside [1500, 3000] ...")
        with pytest.raises(RuntimeError, match="Product"):
            validate_products(sample_products_df)

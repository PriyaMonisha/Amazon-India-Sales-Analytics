import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# TestSlugFromSubcategory (6 parametrized tests)
# ---------------------------------------------------------------------------

@pytest.mark.models
class TestSlugFromSubcategory:
    @pytest.mark.parametrize("raw,expected", [
        ("Smartphones",           "smartphones"),
        ("Mobile Phones/Tablets", "mobile_phones_tablets"),
        ("Smart TVs",             "smart_tvs"),
        ("Audio/Video",           "audio_video"),
        ("Home & Kitchen",        "home_kitchen"),         # & collapsed into single _
        ("Books, Music & Games",  "books_music_games"),   # consecutive non-alphanum → one _
        ("smartphones",           "smartphones"),          # idempotent
    ])
    def test_slug_from_subcategory(self, raw, expected):
        from src.models.forecasting import slug_from_subcategory
        assert slug_from_subcategory(raw) == expected


# ---------------------------------------------------------------------------
# TestDetectAnomalies (7 tests)
# ---------------------------------------------------------------------------

@pytest.mark.models
class TestDetectAnomalies:
    def _make_df(self, final=500.0, mrp=1000.0, days=3.0, is_return=0):
        return pd.DataFrame([{
            "final_amount_inr": final,
            "mrp_inr":          mrp,
            "delivery_days":    days,
            "is_return":        is_return,
        }])

    def test_discount_pct_correct(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        result = detect_anomalies(self._make_df(final=800.0, mrp=1000.0), model, scaler)
        # discount_pct = 1 - 800/1000 = 0.2 — computed and stored in result by detect_anomalies
        assert result["discount_pct"].iloc[0] == pytest.approx(0.2, abs=0.001)

    def test_discount_pct_clips_to_zero(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        # final > mrp → clip(1 - 1500/1000, 0, 1) = clip(-0.5, 0, 1) = 0.0
        result = detect_anomalies(self._make_df(final=1500.0, mrp=1000.0), model, scaler)
        assert result["discount_pct"].iloc[0] == pytest.approx(0.0, abs=0.001)

    def test_discount_pct_clips_to_one(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        # final=0, mrp=1000 → clip(1 - 0/1000, 0, 1) = 1.0
        result = detect_anomalies(self._make_df(final=0.0, mrp=1000.0), model, scaler)
        assert result["discount_pct"].iloc[0] == pytest.approx(1.0, abs=0.001)

    def test_is_anomaly_is_bool_dtype(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        result = detect_anomalies(self._make_df(), model, scaler)
        assert pd.api.types.is_bool_dtype(result["is_anomaly"]), \
            f"Expected bool dtype, got {result['is_anomaly'].dtype}"

    def test_anomaly_score_is_float(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        result = detect_anomalies(self._make_df(), model, scaler)
        score = float(result["anomaly_score"].iloc[0])
        assert isinstance(score, float)
        # decision_function returns continuous values — NOT class labels {-1, 1}
        assert score not in (-1.0, 1.0), \
            f"anomaly_score={score} looks like a class label, not a continuous score"
        assert -1.0 < score < 1.0, \
            f"anomaly_score={score} outside expected IsolationForest range (-1, 1)"

    def test_is_returned_column_alias(self, trained_anomaly_bundle):
        # anomaly.py:176-177 maps is_returned → is_return when is_return absent
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        df = pd.DataFrame([{
            "final_amount_inr": 500.0,
            "mrp_inr":          1000.0,
            "delivery_days":    3.0,
            "is_returned":      0,  # alias — not is_return
        }])
        result = detect_anomalies(df, model, scaler)
        assert "is_anomaly" in result.columns

    def test_scaler_called_before_model(self, trained_anomaly_bundle):
        from src.models.anomaly import detect_anomalies
        model, scaler = trained_anomaly_bundle
        call_order    = []
        real_transform = scaler.transform
        real_predict   = model.predict
        real_decision  = model.decision_function
        mock_s = MagicMock(wraps=scaler)
        mock_m = MagicMock(wraps=model)
        mock_s.transform         = lambda X: (call_order.append("scaler"),   real_transform(X))[1]
        mock_m.predict           = lambda X: (call_order.append("predict"),  real_predict(X))[1]
        mock_m.decision_function = lambda X: (call_order.append("decision"), real_decision(X))[1]

        df = pd.DataFrame([{"final_amount_inr": 500.0, "mrp_inr": 1000.0,
                             "delivery_days": 3.0, "is_return": 0}])
        detect_anomalies(df, mock_m, mock_s)

        assert "scaler" in call_order, f"scaler.transform never called. order={call_order}"
        model_calls = [i for i, x in enumerate(call_order) if x in ("predict", "decision")]
        assert model_calls, f"model never called. order={call_order}"
        # call_order.index("scaler") returns FIRST scaler call — correct even if called multiple times
        assert all(call_order.index("scaler") < m for m in model_calls), \
            f"scaler must be called before model. order={call_order}"


# ---------------------------------------------------------------------------
# TestLoadChurnModel (3 tests)
# ---------------------------------------------------------------------------

@pytest.mark.models
class TestLoadChurnModel:
    def test_raises_when_no_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.churn.ARTIFACTS_DIR", tmp_path)
        (tmp_path / "models").mkdir()
        from src.models.churn import load_churn_model
        with pytest.raises((FileNotFoundError, Exception)):
            load_churn_model()

    def test_bundle_keys_contract(self):
        from src.models.churn import ChurnModelBundle
        assert set(ChurnModelBundle.__annotations__.keys()) == {
            "model", "explainer", "encoder", "feature_names",
            "categorical_features", "threshold", "churn_rate_train", "churn_rate_test",
        }

    def test_all_features_length(self):
        from src.models.churn import ALL_FEATURES
        assert len(ALL_FEATURES) == 14


# ---------------------------------------------------------------------------
# TestLoadForecastModel (2 tests)
# ---------------------------------------------------------------------------

@pytest.mark.models
class TestLoadForecastModel:
    def test_raises_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.forecasting.ARTIFACTS_DIR", tmp_path)
        (tmp_path / "models").mkdir()
        from src.models.forecasting import load_forecast_model
        with pytest.raises(FileNotFoundError):
            load_forecast_model("Smartphones")

    def test_slug_conversion_in_load(self, tmp_path, monkeypatch):
        # Capital "Smartphones" → slug_from_subcategory → "smartphones" → finds the file
        monkeypatch.setattr("src.models.forecasting.ARTIFACTS_DIR", tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "forecast_smartphones.json").write_text('{"placeholder": true}')
        mock_prophet = MagicMock()
        from src.models.forecasting import load_forecast_model
        with patch("src.models.forecasting.model_from_json", return_value=mock_prophet):
            result = load_forecast_model("Smartphones")
        assert result is mock_prophet  # proves slug conversion found the file


# ---------------------------------------------------------------------------
# TestTrainAnomalyModel (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.models
class TestTrainAnomalyModel:
    def test_required_keys(self, anomaly_mocks):
        # Import inside function — guards against future module-level I/O in anomaly.py
        from src.models.anomaly import train_anomaly_model
        synthetic, tmp_path, mock_engine = anomaly_mocks
        with patch("pandas.read_sql", return_value=synthetic), \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_artifact"):
            bundle = train_anomaly_model(mock_engine)
        assert {"model", "scaler", "anomaly_rate", "n_anomalies", "feature_names"}.issubset(bundle)

    def test_anomaly_rate_in_range(self, anomaly_mocks):
        from src.models.anomaly import train_anomaly_model
        synthetic, tmp_path, mock_engine = anomaly_mocks
        with patch("pandas.read_sql", return_value=synthetic), \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_artifact"):
            bundle = train_anomaly_model(mock_engine)
        assert 0.0 < bundle["anomaly_rate"] < 0.1, \
            f"anomaly_rate={bundle['anomaly_rate']} outside expected range (0, 0.1)"

    def test_artifacts_saved_to_disk(self, anomaly_mocks):
        from src.models.anomaly import train_anomaly_model
        synthetic, tmp_path, mock_engine = anomaly_mocks
        with patch("pandas.read_sql", return_value=synthetic), \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_artifact"):
            train_anomaly_model(mock_engine)
        model_path  = tmp_path / "models" / "anomaly_model.pkl"
        scaler_path = tmp_path / "models" / "anomaly_scaler.pkl"
        assert model_path.exists(), \
            f"anomaly_model.pkl not found. Contents: {list((tmp_path / 'models').iterdir())}"
        assert scaler_path.exists(), \
            f"anomaly_scaler.pkl not found. Contents: {list((tmp_path / 'models').iterdir())}"

    def test_feature_names(self, anomaly_mocks):
        from src.models.anomaly import train_anomaly_model
        synthetic, tmp_path, mock_engine = anomaly_mocks
        with patch("pandas.read_sql", return_value=synthetic), \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_artifact"):
            bundle = train_anomaly_model(mock_engine)
        assert bundle["feature_names"] == [
            "final_amount_inr", "discount_pct", "delivery_days", "is_return",
        ]

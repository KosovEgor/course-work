import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
 
import main
from main import app, compute_features_live, build_prediction_text
 
client = TestClient(app)
 
FEATURES = [
    "Dist_To_Swing_High", "Dist_To_Swing_Low",
    "Order_Block", "MA_20", "MA_50", "RSI", "diff",
]
 

def make_synthetic_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    dates  = pd.date_range("2024-11-01", periods=n, freq="4h")
    close  = 300 + np.cumsum(np.random.randn(n) * 2)
    spread = np.abs(np.random.randn(n)) * 1.5
    return pd.DataFrame({
        "time":   dates,
        "Open":   close + np.random.randn(n) * 0.5,
        "High":   close + spread,
        "Low":    close - spread,
        "Close":  close,
        "Volume": np.random.randint(1000000, 10000000, n).astype(float),
    })

def make_feature_row(**overrides) -> pd.Series:
    defaults = {"Close": 310.5, "RSI": 55.0, "MA_20": 308.0, "MA_50": 305.0}
    defaults.update(overrides)
    return pd.Series(defaults)

def mock_cb(pred: int = 1) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = np.array([pred])
    return m

class TestComputeFeaturesLive:
 
    def test_returns_dataframe(self):
        result = compute_features_live(make_synthetic_df())
        assert isinstance(result, pd.DataFrame)
 
    def test_has_all_feature_columns(self):
        df = compute_features_live(make_synthetic_df())
        for col in FEATURES:
            assert col in df.columns, f"Missing column: {col}"
 
    def test_no_nulls_in_features(self):
        df = compute_features_live(make_synthetic_df())
        assert df[FEATURES].isnull().sum().sum() == 0
 

class TestBuildPredictionText:
 
    def test_buy_signal_in_text(self):
        text = build_prediction_text(make_feature_row(), "КУПИТЬ", "PDT")
        assert "КУПИТЬ" in text
 
    def test_sell_signal_in_text(self):
        text = build_prediction_text(make_feature_row(), "ПРОДАТЬ", "PDT")
        assert "ПРОДАТЬ" in text

    def test_price_appears_in_text(self):
        text = build_prediction_text(make_feature_row(Close=310.5), "КУПИТЬ", "PDT")
        assert "310.50" in text

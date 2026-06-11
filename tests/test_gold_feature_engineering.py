import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.processing.gold_feature_engineering import create_gold_features


class TestGoldFeatureEngineering(unittest.TestCase):
    def test_create_gold_features_includes_rsi_14_and_bounds(self):
        n = 60
        close = np.linspace(100.0, 130.0, n)
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * n,
                "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
                "close_price": close,
                "open_price": close,
                "high_price": close,
                "low_price": close,
                "volume": np.full(n, 1000),
            }
        )
        out = create_gold_features(df)
        self.assertIn("rsi_14", out.columns)
        rsi = pd.to_numeric(out["rsi_14"], errors="coerce").dropna()
        self.assertGreater(len(rsi), 10)
        self.assertGreaterEqual(float(rsi.min()), 0.0)
        self.assertLessEqual(float(rsi.max()), 100.0)

    def test_create_gold_features_includes_macd_columns(self):
        n = 80
        close = np.linspace(100.0, 140.0, n)
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * n,
                "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
                "close_price": close,
                "open_price": close,
                "high_price": close,
                "low_price": close,
                "volume": np.full(n, 1000),
            }
        )
        out = create_gold_features(df)
        for c in ("macd_line", "macd_signal", "macd_hist"):
            self.assertIn(c, out.columns)
        # After EMA warmup, values should be finite.
        tail = out[["macd_line", "macd_signal", "macd_hist"]].tail(10).apply(pd.to_numeric, errors="coerce")
        self.assertTrue(np.isfinite(tail.to_numpy()).all())

    def test_rsi_flat_series_is_neutral_after_window(self):
        n = 40
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * n,
                "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
                "close_price": close,
                "open_price": close,
                "high_price": close,
                "low_price": close,
                "volume": np.full(n, 1000),
            }
        )
        out = create_gold_features(df)
        tail = pd.to_numeric(out["rsi_14"], errors="coerce").dropna().tail(5)
        self.assertGreater(len(tail), 0)
        self.assertTrue(np.allclose(tail.to_numpy(), 50.0, atol=1e-9))

    def test_rsi_rising_series_is_high_after_window(self):
        n = 60
        close = np.linspace(100.0, 160.0, n)
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * n,
                "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
                "close_price": close,
                "open_price": close,
                "high_price": close,
                "low_price": close,
                "volume": np.full(n, 1000),
            }
        )
        out = create_gold_features(df)
        rsi_last = float(pd.to_numeric(out["rsi_14"], errors="coerce").dropna().iloc[-1])
        self.assertGreaterEqual(rsi_last, 99.0)


if __name__ == "__main__":
    unittest.main()


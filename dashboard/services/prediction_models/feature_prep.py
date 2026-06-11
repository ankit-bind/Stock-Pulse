from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import os

from dashboard.services.portfolio_service import _daily_return_series_prefer_varying_close
from etl.processing.indicators.macd import compute_macd

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "momentum_10",
    "volatility_10",
    "sma_ratio",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
)


def _compute_rsi_14_fallback(close: pd.Series) -> pd.Series:
    """Fallback RSI(14) for legacy inputs lacking Gold's rsi_14 (kept consistent with ETL)."""
    rsi_window = 14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=rsi_window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.where(~both_zero, 50.0)
    return rsi.clip(0.0, 100.0)


def create_features(df: pd.DataFrame, target_horizon: int = 1) -> pd.DataFrame:
    """
    Build lag-safe features at time t and a future target.

    Args:
        df: DataFrame with OHLCV data.
        target_horizon: Number of days ahead to predict (default=1 for next-day).
                        Options: 1, 5, 20, 60 (1-day, 5-day, 20-day, 3-month).
    
    Target is the future return over the horizon; features only use information available through t.
    """
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
        # One row per calendar day before pct_change — duplicate dates in gold make adjacent
        # returns ~0 and break ML targets + attach_ml_strategy_returns (flat OOS PnL).
        out = out.drop_duplicates(subset=["trade_date"], keep="last")

    # Prefer close-based returns; fall back to gold daily_return when close path is flat (portfolio_service).
    out["daily_return"] = _daily_return_series_prefer_varying_close(out)
    close = pd.to_numeric(out["close_price"], errors="coerce")

    # Future return over the horizon: (close_t+horizon / close_t) - 1
    horizon = max(1, int(target_horizon))
    out["target"] = (close.shift(-horizon) / close) - 1.0
    out["target_horizon"] = horizon  # keep track for diagnostics

    out["ret_1"] = out["daily_return"]
    out["ret_3"] = out["daily_return"].rolling(3, min_periods=3).mean()
    out["ret_5"] = out["daily_return"].rolling(5, min_periods=5).mean()

    base_close = close.shift(10)
    out["momentum_10"] = (close / base_close) - 1.0

    out["volatility_10"] = out["daily_return"].rolling(10, min_periods=10).std()

    sma_20 = pd.to_numeric(out["sma_20"], errors="coerce") if "sma_20" in out.columns else pd.Series(np.nan, index=out.index)
    sma_50 = pd.to_numeric(out["sma_50"], errors="coerce") if "sma_50" in out.columns else pd.Series(np.nan, index=out.index)
    if sma_20.isna().all() or sma_50.isna().all():
        sma_20 = close.rolling(20, min_periods=20).mean()
        sma_50 = close.rolling(50, min_periods=50).mean()
    out["sma_ratio"] = sma_20 / sma_50

    if "rsi_14" in out.columns:
        out["rsi_14"] = pd.to_numeric(out["rsi_14"], errors="coerce")
    else:
        env = (os.getenv("ENV") or "dev").strip().lower()
        if env == "prod":
            raise ValueError("rsi_14 missing in production input")
        logger.info("rsi_14 missing in input; computing fallback RSI(14) in feature_prep")
        out["rsi_14"] = _compute_rsi_14_fallback(close)

    macd_cols = ("macd_line", "macd_signal", "macd_hist")
    if all(c in out.columns for c in macd_cols):
        for c in macd_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    else:
        env = (os.getenv("ENV") or "dev").strip().lower()
        if env == "prod":
            raise ValueError("macd_* missing in production input")
        logger.info("macd_* missing in input; computing fallback MACD in feature_prep")
        macd_df = compute_macd(close)
        out = pd.concat([out, macd_df], axis=1)

    out = out.dropna(subset=list(DEFAULT_FEATURE_COLS) + ["target"]).reset_index(drop=True)
    return out

import numpy as np
import pandas as pd

from etl.processing.indicators.macd import compute_macd

SMA_WINDOW = 20
SMA_50_WINDOW = 50
RSI_WINDOW = 14


def create_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build analytics columns from Silver OHLCV: SMAs, daily return, trend vs SMA-20,
    buy/sell/hold signal from SMA-20 vs SMA-50, and RSI(14).
    """
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["symbol", "trade_date", "close_price"])
    out = out.sort_values(["symbol", "trade_date"])
    out = out.drop_duplicates(subset=["symbol", "trade_date"], keep="last")

    chunks = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.copy()
        g = g.sort_values("trade_date")
        g["daily_return"] = g["close_price"].pct_change()

        # RSI(14) using rolling average gains/losses (causal).
        delta = g["close_price"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(window=RSI_WINDOW, min_periods=RSI_WINDOW).mean()
        avg_loss = loss.rolling(window=RSI_WINDOW, min_periods=RSI_WINDOW).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Edge handling:
        # - no movement (avg_gain=0 and avg_loss=0): RSI = 50 (neutral)
        # - no losses (avg_loss=0): RSI = 100
        # - no gains (avg_gain=0): RSI = 0
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        rsi = rsi.where(avg_gain != 0.0, 0.0)
        both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
        rsi = rsi.where(~both_zero, 50.0)
        g["rsi_14"] = rsi.clip(0.0, 100.0)

        macd_df = compute_macd(g["close_price"])
        g = pd.concat([g, macd_df], axis=1)

        window = min(SMA_WINDOW, max(1, len(g)))
        g["sma_20"] = g["close_price"].rolling(window=window, min_periods=1).mean()
        g["sma_50"] = g["close_price"].rolling(
            window=SMA_50_WINDOW, min_periods=1
        ).mean()
        close = g["close_price"].to_numpy()
        sma = g["sma_20"].to_numpy()
        g["trend"] = np.where(close > sma, "up", np.where(close < sma, "down", "flat"))
        g["signal"] = "hold"
        g.loc[g["sma_20"] > g["sma_50"], "signal"] = "buy"
        g.loc[g["sma_20"] < g["sma_50"], "signal"] = "sell"
        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)

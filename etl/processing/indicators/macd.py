from __future__ import annotations

import pandas as pd

FAST = 12
SLOW = 26
SIGNAL = 9


def compute_macd(close: pd.Series) -> pd.DataFrame:
    """
    Compute MACD (line, signal, histogram) from close prices.

    Uses exponential moving averages (EMA) and is causal (no future leakage).
    """
    close = pd.to_numeric(close, errors="coerce")

    ema_fast = close.ewm(span=FAST, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    )


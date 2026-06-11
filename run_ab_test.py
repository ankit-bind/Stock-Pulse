import math
from typing import Any

import pandas as pd

from dashboard.services.data_service import fetch_stock_data
from dashboard.services.prediction_models.feature_prep import create_features, DEFAULT_FEATURE_COLS
from dashboard.services.prediction_models.walk_forward import walk_forward_predict
from dashboard.services.prediction_models.random_forest import train_rf, predict_rf
from dashboard.services.prediction_models.model_evaluator import evaluate_model
from dashboard.services.prediction_models.ml_backtest import (
    attach_ml_strategy_returns,
    sharpe_daily,
    cagr_from_cum,
    max_drawdown_from_cum,
    realized_turnover,
)

BASE_COLS = tuple(
    c for c in DEFAULT_FEATURE_COLS if c not in ("macd_line", "macd_signal", "macd_hist")
)


def _row_count_and_range(df: pd.DataFrame) -> tuple[int, str]:
    if df is None or df.empty or "trade_date" not in df.columns:
        return (0, "n/a")
    td = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
    if td.empty:
        return (len(df), "n/a")
    return (len(df), f"{td.min().date()} to {td.max().date()}")


def run(symbol: str, feature_cols: tuple[str, ...], train_size: float = 0.7) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        raise ValueError("no data fetched from gold layer (empty dataframe)")
    feats = create_features(df)

    raw_n, raw_rng = _row_count_and_range(df)
    feat_n, feat_rng = _row_count_and_range(feats)
    print(f"{symbol}: raw_rows={raw_n} ({raw_rng}) | feature_rows={feat_n} ({feat_rng})")

    if len(feats) < 3:
        raise ValueError(f"not enough feature rows after preprocessing ({len(feats)})")

    _, oos, _, _ = walk_forward_predict(
        feats, train_rf, predict_rf, feature_cols, train_size=train_size
    )

    m = evaluate_model(oos)

    bt = attach_ml_strategy_returns(
        oos,
        signal_mode="long_only",
        threshold_mode="static",
        prediction_threshold=0.0,
        transaction_cost_pct=0.001,
        include_rf=False,
    )

    sh = sharpe_daily(bt["strategy_return"])
    sharpe_ann = sh * math.sqrt(252) if sh == sh else float("nan")

    return oos, {
        "n_oos": len(oos),
        "ic_s": m["ic_spearman"],
        "sharpe": sharpe_ann,
        "cagr": cagr_from_cum(bt["cum_strategy_return"]),
        "max_dd": max_drawdown_from_cum(bt["cum_strategy_return"]),
        "turnover": realized_turnover(bt),
    }

def ab(symbol: str) -> None:
    try:
        oos_a, res_a = run(symbol, BASE_COLS)
        oos_b, res_b = run(symbol, DEFAULT_FEATURE_COLS)
    except ValueError as e:
        print(f"\n===== {symbol} =====")
        print(f"SKIP: {e}")
        return

    # Align by timestamp, not by positional index, to ensure a fair comparison.
    a_td = pd.to_datetime(oos_a.get("trade_date"), errors="coerce")
    b_td = pd.to_datetime(oos_b.get("trade_date"), errors="coerce")
    common_dates = set(a_td.dropna()).intersection(set(b_td.dropna()))
    oos_a_aligned = oos_a.loc[a_td.isin(common_dates)].copy()
    oos_b_aligned = oos_b.loc[b_td.isin(common_dates)].copy()

    # If duplicates or ordering differences exist, inner-join on exact timestamps.
    if len(oos_a_aligned) != len(oos_b_aligned):
        oa = oos_a.copy()
        ob = oos_b.copy()
        oa["_td"] = pd.to_datetime(oa["trade_date"], errors="coerce")
        ob["_td"] = pd.to_datetime(ob["trade_date"], errors="coerce")
        merged = oa.merge(ob, on="_td", how="inner", suffixes=("_a", "_b"))
        oos_a_aligned = merged.filter(regex=r"_a$").rename(columns=lambda c: c[:-2])
        oos_b_aligned = merged.filter(regex=r"_b$").rename(columns=lambda c: c[:-2])

    oos_size = len(oos_a_aligned)
    assert len(oos_a_aligned) == len(oos_b_aligned)
    assert oos_size > 100

    print(f"\n===== {symbol} =====")
    print("WITHOUT MACD:", res_a)
    print("WITH MACD:", res_b)
    print("OOS size:", oos_size)
    print("DELTA Sharpe:", res_b["sharpe"] - res_a["sharpe"])
    print("DELTA IC:", res_b["ic_s"] - res_a["ic_s"])

for sym in ["AAPL", "INFY.NS", "HCLTECH.NS"]:
    ab(sym)
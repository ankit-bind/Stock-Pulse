import math
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.styles.chart_theme import apply_theme
import streamlit as st
from plotly import io as pio

from dashboard.services.data_service import fetch_symbols, fetch_stock_data
from dashboard.services.portfolio_service import build_symbol_returns_panel, compute_portfolio

TRADING_DAYS_PER_YEAR = 252

def _chart_explanation(text: str) -> None:
    st.info(f"**Chart Guide:** {text}")

def _chart_data_table(df: pd.DataFrame, n: int = 5) -> None:
    st.caption("Recent Data Points")
    st.dataframe(df.tail(n).style.format({
        "close_price": "{:.2f}",
        "daily_return": "{:.4f}",
        "sma_short": "{:.2f}",
        "sma_long": "{:.2f}",
        "rsi_14": "{:.1f}",
    }), use_container_width=True)

def _chart_insights(**metrics) -> None:
    st.markdown("### Key Insights")
    cols = st.columns(min(len(metrics), 4))
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            st.metric(label, value)

def _pipeline_df_view(
    df: pd.DataFrame,
    start_date,
    end_date,
    sma_short_n: int,
    sma_long_n: int,
    transaction_cost_pct: float = 0.0,
    rf_annual: float = 0.0,
    include_rf: bool = False,
) -> pd.DataFrame:
    """Sort, compute SMAs on full history, slice to date range, add returns and trade markers."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    out["daily_return"] = close.pct_change().fillna(0.0)
    out["sma_short"] = close.rolling(sma_short_n, min_periods=1).mean()
    out["sma_long"] = close.rolling(sma_long_n, min_periods=1).mean()

    td = out["trade_date"].dt.date
    out = out[(td >= start_date) & (td <= end_date)].copy()
    if out.empty:
        return out

    dr = pd.to_numeric(out["daily_return"], errors="coerce").fillna(0.0)
    signal_long = (out["sma_short"] > out["sma_long"]).fillna(False).astype(int)
    out["position"] = signal_long.shift(1).fillna(0).astype(int)
    out["trade_change"] = out["position"].diff().fillna(0).astype(int)
    out["trade"] = out["trade_change"].abs().fillna(0)
    cost_drag = out["trade"] * float(transaction_cost_pct)
    out["cost_drag"] = cost_drag
    rf_daily = float(rf_annual) / TRADING_DAYS_PER_YEAR if include_rf else 0.0
    out["rf_daily"] = rf_daily
    out["strategy_return"] = (
        (out["position"].astype(float) * dr)
        + ((1.0 - out["position"].astype(float)) * rf_daily)
        - cost_drag
    )
    out["cum_market_return"] = (1 + dr).cumprod()
    out["cum_strategy_return"] = (1 + out["strategy_return"]).cumprod()
    return out

def _benchmark_buy_hold(df: pd.DataFrame, start_date, end_date) -> Optional[dict]:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    dr_all = close.pct_change().fillna(0.0)
    td = out["trade_date"].dt.date
    out = out[(td >= start_date) & (td <= end_date)].copy()
    if out.empty:
        return None
    dr = pd.to_numeric(dr_all.loc[out.index], errors="coerce").fillna(0.0)
    cum = (1 + dr).cumprod()
    n = len(out)
    y = n / TRADING_DAYS_PER_YEAR if n else 0.0
    cm_end = float(cum.iloc[-1])
    total = cm_end - 1.0
    cagr = cm_end ** (1.0 / y) - 1.0 if y > 0 and cm_end > 0 else float("nan")
    return {"total": total, "cagr": cagr, "n_days": n}

def _round_trip_returns(df_view: pd.DataFrame) -> list[float]:
    dv = df_view.reset_index(drop=True)
    if dv.empty or "trade_change" not in dv.columns:
        return []
    buys = dv.index[dv["trade_change"] == 1].tolist()
    sells = dv.index[dv["trade_change"] == -1].tolist()
    close = dv["close_price"].astype(float)
    rets: list[float] = []
    si = 0
    for b in buys:
        while si < len(sells) and sells[si] <= b:
            si += 1
        if si >= len(sells):
            break
        s = sells[si]
        cb = float(close.iloc[b])
        cs = float(close.iloc[s])
        if pd.notna(cb) and pd.notna(cs) and cb != 0:
            rets.append(cs / cb - 1.0)
        si += 1
    return rets

def _sharpe_daily(sr: pd.Series) -> float:
    if len(sr) <= 1:
        return float("nan")
    std = float(sr.std(ddof=1))
    if std <= 0 or math.isnan(std):
        return float("nan")
    return float(sr.mean() / std)

def show():
    pio.templates.default = "plotly_dark"
    render_neon_card = st.session_state.get("render_neon_card")

    symbols = fetch_symbols()
    if not symbols:
        st.warning("No symbols found in the database. Run the ETL pipeline to load stock data.")
        with st.expander("How to run the ETL pipeline"):
            st.markdown(
                """
                **Option A (SQL Server):**
                ```bash
                python -m etl.run_pipeline
                ```
                **Option B (SQLite quick demo):**
                1. Set `USE_SQLITE=true` in `configuration/.env`
                2. Run:
                ```bash
                python -m etl.run_pipeline
                ```
                """
            )
        return

    st.sidebar.markdown("### Analysis Settings")
    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        symbols,
        help="Choose which stock to analyze."
    )

    df = fetch_stock_data(selected_symbol)
    if df.empty:
        st.info("No rows for this symbol.")
        return

    df = df.copy()
    if "signal" in df.columns:
        df["signal"] = df["signal"].apply(lambda x: "hold" if pd.isna(x) else str(x).strip().lower())

    dmin = pd.to_datetime(df["trade_date"], errors="coerce").min().date()
    dmax = pd.to_datetime(df["trade_date"], errors="coerce").max().date()

    start_date = st.sidebar.date_input(
        "Start date",
        value=dmin,
        min_value=dmin,
        max_value=dmax,
        help="Analysis start date."
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=dmax,
        min_value=dmin,
        max_value=dmax,
        help="Analysis end date."
    )

    st.sidebar.markdown("#### Strategy Parameters")
    sma_short_n = st.sidebar.slider(
        "Short SMA",
        5, 50, 20,
        help="Short-term moving average. When price crosses above this, it may signal a buy."
    )
    long_lo = max(20, sma_short_n + 1)
    sma_long_n = st.sidebar.slider(
        "Long SMA",
        min_value=long_lo,
        max_value=200,
        value=min(max(50, long_lo), 200),
        help="Long-term moving average. When short SMA crosses above this = BUY signal."
    )
    cost_pct = st.sidebar.number_input(
        "Transaction cost/side",
        min_value=0.0,
        max_value=0.05,
        value=0.001,
        step=0.0005,
        format="%.4f",
        help="Cost per trade (e.g., 0.001 = 0.1%). Includes brokerage + slippage."
    )
    bench_options = [s for s in symbols if s != selected_symbol]
    bench_label = st.sidebar.selectbox(
        "Benchmark",
        ["— None —"] + bench_options,
        help="Compare strategy against another stock."
    )
    benchmark_sym = None if bench_label == "— None —" else bench_label
    include_rf = st.sidebar.checkbox(
        "Include risk-free return",
        value=False,
        help="Earn interest on cash when not holding stocks."
    )
    rf_annual = st.sidebar.number_input(
        "Risk-free rate (annual)",
        min_value=0.0,
        max_value=0.25,
        value=0.05,
        step=0.005,
        format="%.3f",
        disabled=not include_rf,
        help="Risk-free interest rate. 0.05 = 5% per year."
    )

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    df_view = _pipeline_df_view(
        df,
        start_date,
        end_date,
        sma_short_n,
        sma_long_n,
        transaction_cost_pct=cost_pct,
        rf_annual=rf_annual,
        include_rf=include_rf,
    )

    if df_view.empty:
        st.info("No rows in the selected date range.")
        return
        
    st.markdown(
        f"<div style='font-size:1.4rem; font-weight:600; color:#e2e8f0; "
        f"letter-spacing:0.01em; margin-bottom:0.5rem;'>[Chart] "
        f"<span style='color:#00d4ff;'>{selected_symbol}</span> "
        f"<span style='color:#94a3b8; font-weight:400;'>Analysis</span></div>",
        unsafe_allow_html=True,
    )

    # Beginner help
    with st.expander("How to read this page (click to learn)", expanded=False):
        st.markdown("""
        **Key Metrics:**
        - **Price Δ**: Latest close price + change from previous day
        - **RSI(14)**: 0-100 scale. <30 = oversold (may bounce up), >70 = overbought (may fall)
        - **SMA Signal**: LONG = buy, FLAT = sell, NEUTRAL = hold
        - **Strategy CAGR**: Annual return of the SMA crossover strategy
        
        **Charts:**
        - **Candlestick**: Green = price up, Red = price down
        - **Blue line**: Short SMA (fast, 20 days)
        - **Orange line**: Long SMA (slow, 50 days)
        - **Green triangles**: BUY signals (short crossed above long)
        - **Red triangles**: SELL signals (short crossed below long)
        
        **Strategy:**
        - Buys when short SMA crosses above long SMA
        - Sells when short SMA crosses below long SMA
        - Transaction costs reduce returns
        """)

    # Metrics for KPI Cards
    latest = df_view.iloc[-1]
    prev = df_view.iloc[-2] if len(df_view) > 1 else latest
    
    price = latest.get("close_price", 0.0)
    prev_price = prev.get("close_price", 0.0)
    price_delta = ((price / prev_price) - 1.0) * 100 if prev_price else 0.0
    price_str = f"₹{price:,.2f} ({price_delta:+.2f}%)"
    
    vol = latest.get("volume", 0)
    vol_str = f"{vol:,.0f}" if pd.notna(vol) else "N/A"
    
    rsi = latest.get("rsi_14", float("nan"))
    rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"
    
    ss = latest["sma_short"]
    sl = latest["sma_long"]
    signal_str = "LONG" if ss > sl else ("FLAT" if ss < sl else "NEUTRAL")
    
    n_days = len(df_view)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days else 0.0
    cs_end = float(df_view["cum_strategy_return"].iloc[-1])
    cagr_strategy = (cs_end ** (1.0 / years) - 1.0) if (years > 0 and cs_end > 0) else float("nan")
    cagr_str = f"{cagr_strategy:.2%}" if pd.notna(cagr_strategy) else "—"

    # KPI Cards
    if render_neon_card:
        cards_html = (
            '<div class="kpi-grid">\n'
            f'{render_neon_card("Price Δ", price_str)}'
            f'{render_neon_card("Volume", vol_str)}'
            f'{render_neon_card("RSI(14)", rsi_str)}'
            f'{render_neon_card("SMA Signal", signal_str)}'
            f'{render_neon_card("Strategy CAGR", cagr_str)}'
            '</div>'
        )
        st.markdown(cards_html, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Overview", "Strategy", "Portfolio"])

    with tab1:
        st.subheader("Price Action & Technicals")
        # Main Candlestick Chart
        from plotly.subplots import make_subplots
        
        has_vol = "volume" in df_view.columns
        has_rsi = "rsi_14" in df_view.columns
        
        rows = 1 + int(has_vol) + int(has_rsi)
        row_heights = [0.6]
        if has_vol: row_heights.append(0.2)
        if has_rsi: row_heights.append(0.2)
        
        titles = ['Price (OHLC)']
        if has_vol: titles.append('Volume')
        if has_rsi: titles.append('RSI (14)')

        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.12, row_heights=row_heights,
            subplot_titles=titles
        )
        
        if all(c in df_view.columns for c in ["open_price", "high_price", "low_price", "close_price"]):
            fig.add_trace(go.Candlestick(
                x=df_view['trade_date'], open=df_view['open_price'], high=df_view['high_price'],
                low=df_view['low_price'], close=df_view['close_price'], name="OHLC",
                increasing_fillcolor="#10b981", decreasing_fillcolor="#ef4444",
                increasing_line_color="rgba(255,255,255,0.4)", decreasing_line_color="rgba(255,255,255,0.4)"
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_view["trade_date"], y=df_view["close_price"], mode="lines", name="Close"), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=df_view["trade_date"], y=df_view["sma_short"], mode="lines", name=f"SMA {sma_short_n}", line=dict(color="#3498db")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view["trade_date"], y=df_view["sma_long"], mode="lines", name=f"SMA {sma_long_n}", line=dict(color="#e67e22")), row=1, col=1)

        buy_df = df_view[df_view["trade_change"] == 1]
        sell_df = df_view[df_view["trade_change"] == -1]

        fig.add_trace(go.Scatter(x=buy_df["trade_date"], y=buy_df["close_price"], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=14, color="#2ecc71", line=dict(width=1, color="white"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_df["trade_date"], y=sell_df["close_price"], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=14, color="#e74c3c", line=dict(width=1, color="white"))), row=1, col=1)

        curr_row = 2
        if has_vol:
            colors = ['#10b981' if row['daily_return'] >= 0 else '#ef4444' for i, row in df_view.iterrows()]
            fig.add_trace(go.Bar(x=df_view["trade_date"], y=df_view["volume"], marker_color=colors, marker_line_width=0, name="Volume"), row=curr_row, col=1)
            curr_row += 1
            
        if has_rsi:
            rsi_df = df_view.dropna(subset=["rsi_14"]).copy()
            fig.add_trace(go.Scatter(x=rsi_df["trade_date"], y=rsi_df["rsi_14"], mode="lines", name="RSI", line=dict(color="#9b59b6")), row=curr_row, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.10, line_width=0, row=curr_row, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.10, line_width=0, row=curr_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=curr_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=curr_row, col=1)
            fig.update_yaxes(range=[0, 100], row=curr_row, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        if has_vol: fig.update_yaxes(title_text="Volume", row=2, col=1)
        if has_rsi: fig.update_yaxes(title_text="RSI", row=rows, col=1)

        fig.update_layout(height=1000 if rows == 3 else 700, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False)
        st.markdown('''<div class="chart-label">Price &middot; Volume &middot; RSI</div>''', unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

        # Chart additions
        _chart_explanation(
            "Candlestick colors: Green = Price up, Red = Price down. "
            "Blue line = Short SMA, Orange line = Long SMA. "
            "Triangles = Buy/Sell signals from crossover."
        )
        _chart_data_table(df_view)
        _chart_insights(
            Latest_Price=f"₹{latest.get('close_price', 0):,.2f}",
            Latest_RSI=f"{rsi:.1f}",
            SMA_Signal=signal_str,
        )

    with tab2:
        st.subheader("Strategy vs Market (Cumulative)")
        perf = df_view[["trade_date", "cum_market_return", "cum_strategy_return"]].copy()
        perf["market_return"] = perf["cum_market_return"] - 1.0
        perf["strategy_return"] = perf["cum_strategy_return"] - 1.0
        fig_perf = px.line(
            perf, x="trade_date", y=["market_return", "strategy_return"],
            labels={"value": "Return", "variable": "Series"},
            color_discrete_sequence=["#e74c3c", "#3498db"]
        )
        fig_perf.update_layout(hovermode="x unified", legend_title_text="", yaxis_tickformat=".0%")
        st.markdown('''<div class="chart-label">Strategy vs Market &middot; Cumulative Return</div>''', unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig_perf), use_container_width=True)

        _chart_explanation(
            "Red line = Buy & Hold market return. Blue line = SMA crossover strategy return. "
            "Strategy uses lagged signals (no lookahead bias). "
            "Costs are deducted per trade change."
        )
        _chart_data_table(perf)

        st.subheader("Performance & Risk")
        total_return = float(df_view["cum_strategy_return"].iloc[-1] - 1.0)
        market_return = float(df_view["cum_market_return"].iloc[-1] - 1.0)
        cm_end = float(df_view["cum_market_return"].iloc[-1])
        cagr_market = (cm_end ** (1.0 / years) - 1.0) if (years > 0 and cm_end > 0) else float("nan")
        alpha_cagr = float(cagr_strategy - cagr_market) if pd.notna(cagr_strategy) and pd.notna(cagr_market) else float("nan")
        exposure_pct = float(df_view["position"].mean()) if n_days else float("nan")
        sharpe_daily = _sharpe_daily(df_view["strategy_return"])
        sharpe_annual = sharpe_daily * (TRADING_DAYS_PER_YEAR ** 0.5) if pd.notna(sharpe_daily) else float("nan")
        cum = df_view["cum_strategy_return"]
        drawdown = (cum / cum.cummax()) - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")
        
        trip_rets = _round_trip_returns(df_view)
        n_trips = len(trip_rets)
        win_rate = float(sum(1 for r in trip_rets if r > 0) / n_trips) if n_trips else float("nan")
        avg_trade = float(sum(trip_rets) / n_trips) if n_trips else float("nan")
        n_buys = int((df_view["trade_change"] == 1).sum())
        n_sells = int((df_view["trade_change"] == -1).sum())

        if render_neon_card:
            perf_html = (
                '<div class="kpi-grid">\n'
                f'{render_neon_card("Strategy Return", f"{total_return:.2%}")}'
                f'{render_neon_card("Market Return", f"{market_return:.2%}")}'
                f'{render_neon_card("Excess Return", f"{(total_return - market_return):.2%}")}'
                f'{render_neon_card("Alpha (ann.)", f"{alpha_cagr:.2%}" if pd.notna(alpha_cagr) else "—")}'
                f'{render_neon_card("Sharpe (ann.)", f"{sharpe_annual:.2f}" if pd.notna(sharpe_annual) else "—")}'
                f'{render_neon_card("Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "—")}'
                f'{render_neon_card("Time in Market", f"{exposure_pct:.1%}" if pd.notna(exposure_pct) else "—")}'
                f'{render_neon_card("Win Rate", f"{win_rate:.1%}" if pd.notna(win_rate) else "—")}'
                '</div>'
            )
            st.markdown(perf_html, unsafe_allow_html=True)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy Return", f"{total_return:.2%}")
            c2.metric("Market Return", f"{market_return:.2%}")
            c3.metric("Excess Return", f"{(total_return - market_return):.2%}")
            c4.metric("Alpha (ann.)", f"{alpha_cagr:.2%}" if pd.notna(alpha_cagr) else "—")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe (ann.)", f"{sharpe_annual:.2f}" if pd.notna(sharpe_annual) else "—")
            c2.metric("Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "—")
            c3.metric("Time in Market", f"{exposure_pct:.1%}" if pd.notna(exposure_pct) else "—")
            c4.metric("Win Rate", f"{win_rate:.1%}" if pd.notna(win_rate) else "—")

        _chart_insights(
            Strategy_Return=f"{total_return:.2%}",
            Market_Return=f"{market_return:.2%}",
            Alpha_CAGR=f"{alpha_cagr:.2%}" if pd.notna(alpha_cagr) else "—",
            Win_Rate=f"{win_rate:.1%}" if pd.notna(win_rate) else "—",
        )
        _chart_insights(
            Total_Trades=n_buys + n_sells,
            Buys=n_buys,
            Sells=n_sells,
        )

        st.subheader("Cross-sectional Comparison")
        compare_syms = st.multiselect("Compare with", symbols, default=[])
        if compare_syms:
            rows = []
            for sym in compare_syms:
                raw = fetch_stock_data(sym)
                if not raw.empty:
                    dv = _pipeline_df_view(raw, start_date, end_date, sma_short_n, sma_long_n, cost_pct, rf_annual, include_rf)
                    if not dv.empty:
                        n_d = len(dv)
                        y = n_d / TRADING_DAYS_PER_YEAR if n_d else 0.0
                        cs = float(dv["cum_strategy_return"].iloc[-1])
                        st_tot = cs - 1.0
                        cg_s = cs ** (1.0 / y) - 1.0 if y > 0 and cs > 0 else float("nan")
                        rows.append({"Symbol": sym, "Strategy CAGR": cg_s, "Total Return": st_tot})
            if rows:
                comp_df = pd.DataFrame(rows)
                styled_comp = (comp_df.style.format({"Strategy CAGR": "{:.2%}", "Total Return": "{:.2%}"})
                     .map(lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Strategy CAGR", "Total Return"])
                     .set_properties(**{"font-family": "JetBrains Mono"}))
                st.dataframe(styled_comp, use_container_width=True)
                st.download_button(label="Download Comparison CSV", data=comp_df.to_csv(index=False).encode('utf-8'), file_name='comparison.csv', mime='text/csv', key='stock_comp_dl')

    with tab3:
        st.subheader("Portfolio Simulator (Equal-Weight)")
        st.caption("Daily rebalanced equal-weight portfolio using the same date window.")
        default_port = symbols[: min(3, len(symbols))]
        port_syms = st.multiselect("Select stocks for portfolio", options=symbols, default=default_port, key="portfolio_select")
        
        if len(port_syms) < 2:
            st.info("Select at least 2 stocks to build a portfolio.")
        else:
            df_panel = build_symbol_returns_panel(port_syms, start_date, end_date, fetch_stock_data)
            if df_panel.empty:
                st.warning("No overlapping price history.")
            else:
                pivot, pm = None, None
                try:
                    pivot, pm = compute_portfolio(df_panel, port_syms)
                except Exception as e:
                    st.warning(str(e))
                    
                if pivot is not None and pm is not None and not pivot.empty:
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=pivot.index, y=pivot["total_return"] * 100.0, mode="lines", name="Portfolio", line=dict(color="#f1c40f")))
                    fig_p.update_layout(title="Portfolio Cumulative Return", hovermode="x unified", yaxis_title="Return (%)")
                    st.plotly_chart(apply_theme(fig_p), use_container_width=True)

                    _chart_explanation(
                        "Equal-weight portfolio rebalanced daily. Each stock contributes equally. "
                        "Portfolio return is the average of constituent daily returns."
                    )
                    _chart_data_table(pivot.reset_index().rename(columns={"index": "trade_date"}), n=5)
                    _chart_insights(
                        Portfolio_CAGR=f"{pm['cagr']:.2%}" if pd.notna(pm["cagr"]) else "—",
                        Max_Drawdown=f"{pm['max_dd']:.2%}" if pd.notna(pm["max_dd"]) else "—",
                        Sharpe_Annual=f"{pm['sharpe']:.2f}" if pd.notna(pm["sharpe"]) else "—",
                    )
                    
                    if render_neon_card:
                        _p_cagr = f"{pm['cagr']:.2%}" if pd.notna(pm["cagr"]) else "—"
                        _p_dd = f"{pm['max_dd']:.2%}" if pd.notna(pm["max_dd"]) else "—"
                        _p_sharpe = f"{pm['sharpe']:.2f}" if pd.notna(pm["sharpe"]) else "—"
                        port_html = (
                            '<div class="kpi-grid">\n'
                            + render_neon_card("Portfolio CAGR", _p_cagr)
                            + render_neon_card("Max Drawdown", _p_dd)
                            + render_neon_card("Sharpe (ann.)", _p_sharpe)
                            + '</div>'
                        )
                        st.markdown(port_html, unsafe_allow_html=True)
                    else:
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Portfolio CAGR", f"{pm['cagr']:.2%}" if pd.notna(pm["cagr"]) else "—")
                        p2.metric("Max Drawdown", f"{pm['max_dd']:.2%}" if pd.notna(pm["max_dd"]) else "—")
                        p3.metric("Sharpe (ann.)", f"{pm['sharpe']:.2f}" if pd.notna(pm["sharpe"]) else "—")

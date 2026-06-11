# Stock-Pulse

> AI-powered Indian stock prediction dashboard with ML walk-forward backtesting, technical analysis, and beginner-friendly controls.

Stock-Pulse is an end-to-end quantitative research and ML portfolio system for Indian equities. It covers the full pipeline: **ETL -> Features -> ML -> Signals -> Portfolio -> Evaluation**, with causal execution, transaction costs, and cross-sectional books suitable for production-style backtests.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Quick Start](#quick-start)
4. [Database Setup](#database-setup)
5. [Dashboard Views](#dashboard-views)
6. [How to Read Results](#how-to-read-results)
7. [Architecture](#architecture)
8. [Screenshots](#screenshots)
9. [Running Tests](#running-tests)

---

## Features

### Core Capabilities
- **ML Strategy Engine** - Walk-forward validation with RandomForest and optional XGBoost
- **Technical Analysis** - SMA, RSI, MACD, candlestick charts with real-time signals
- **Backtesting** - Chronological validation, next-bar execution, transaction cost modeling
- **Portfolio Construction** - Equal-weight, inverse-volatility, and cross-sectional ML portfolios
- **Risk Metrics** - Sharpe ratio, CAGR, max drawdown, underwater curves, rolling IC

### User Experience
- **Simple Mode** - Beginner-friendly interface with 4 essential controls
- **Advanced Mode** - Full control for quantitative researchers and professionals
- **Quick Presets** - One-click setup: Beginner (Easy), Balanced (Standard), Aggressive (Pro)
- **Inline Help** - Tooltips on every control, chart explanations (What / Why / How) with verdicts and benchmarks
- **Indian Market Focus** - INR currency, Nifty 50 stocks, local brokerage cost assumptions

### Prediction Models
| Model | Description | Use Case |
|-------|-------------|----------|
| RandomForest | Robust ensemble, good default | All skill levels |
| XGBoost | Gradient boosting, fast and accurate | If installed, for pros |

### Prediction Horizons
| Horizon | Timeframe | Best For |
|---------|-----------|----------|
| 1-day | Next day | High frequency, noisy |
| 20-day | 1 month ahead | Balanced signal |
| 60-day | 3 months ahead | Most predictable, beginner recommended |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit, Plotly |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn (RandomForest), optional XGBoost |
| Database | SQL Server (production) or SQLite (demo) |
| ETL | Python with Bronze/Silver/Gold pipeline |
| Visualization | Plotly Dark Theme, Custom Chart Styling |

---

## Quick Start

```bash
# 1. Clone and setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure database (see Database Setup below)
# 3. Run ETL pipeline
python -m etl.run_pipeline

# 4. Launch dashboard
streamlit run dashboard/main.py
```

Open browser at `http://localhost:8501`

---

## Database Setup

### Option A: SQL Server (Production)

1. Ensure SQL Server is running (`localhost\SQLEXPRESS`)
2. Configure `configuration/.env`:
   ```
   DB_SERVER=localhost\SQLEXPRESS
   DB_NAME=StockPulse
   DB_DRIVER=ODBC+Driver+17+for+SQL+Server
   DB_TRUSTED_CONNECTION=yes
   ```
3. Run ETL pipeline: `python -m etl.run_pipeline`

### Option B: SQLite (Quick Demo)

1. Set `USE_SQLITE=true` in `configuration/.env`
2. Run ETL pipeline: `python -m etl.run_pipeline`
3. No SQL Server required - perfect for quick demos and testing

> Configure database paths as required for your environment. Do not commit secrets.

---

## Dashboard Views

### 1. Stock Analysis
- Interactive candlestick charts with SMA overlays
- Buy/Sell signal markers on chart
- RSI, MACD technical indicators
- Strategy vs Buy-and-Hold performance comparison
- Key metrics with Indian market benchmarks

### 2. ML Strategy (Walk-Forward)
- Chronological validation with next-bar execution
- Optional risk-free carry on flat cash
- Head-to-head comparison vs SMA benchmark
- Cross-sectional ML portfolio construction
- Every chart includes: What, Why, How, Verdict, and Benchmarks

### 3. Portfolio Builder
- Multi-stock analysis and comparison
- Correlation heatmaps
- Portfolio-level risk metrics

---

## How to Read Results

### Key Metrics
| Metric | Meaning | Good Range |
|--------|---------|-----------|
| **CAGR** | Annualized return | > 15% beats Nifty 50 |
| **Sharpe** | Risk-adjusted return | > 1.0 is good, > 1.5 is excellent |
| **Max DD** | Worst peak-to-trough loss | < 20% is conservative, < 30% is manageable |
| **IC** | Information Coefficient (prediction accuracy) | > 0.05 is good, > 0.10 is strong |
| **Turnover** | Trading frequency | Lower = less cost drag |

### Benchmarks (Indian Market)
- Nifty 50 long-term CAGR: ~12%
- Nifty 50 typical max drawdown: ~30-35%
- Good ML strategy: CAGR > 15%, Sharpe > 1.0, Max DD < 25%

### Reading Charts
Every chart in the ML Strategy view includes:
- **What** - What the chart shows
- **Why** - Why it matters for decision making
- **How** - How to interpret the results
- **Verdict** - Green/Yellow/Red indicator of quality
- **Benchmarks** - Comparison against Indian market standards

---

## Architecture

![Stock-Pulse Architecture](docs/architecture_diagram.png)

**Architecture Overview:**
1. **Data Sources** - CSV files with OHLCV data for Indian stocks
2. **ETL Pipeline** - Bronze -> Silver -> Gold layers with standardization
3. **Database** - SQL Server (production) or SQLite (local demo)
4. **Feature Engineering** - Technical indicators (RSI, MACD, SMA), causal targets
5. **ML Prediction** - RandomForest / XGBoost with walk-forward validation
6. **Portfolio** - Equal-weight, inverse-vol, cost-aware execution, beta-neutral options
7. **Dashboard** - Streamlit with Plotly visualizations, simple and advanced modes

---

## Screenshots

```markdown
![Stock Analysis](docs/screenshots/stock_analysis.png)
![ML Strategy](docs/screenshots/ml_strategy.png)
![Portfolio](docs/screenshots/portfolio.png)
```

---

## Running Tests

```bash
# Run prediction model tests
python -m unittest tests.test_prediction_models -v

# Run feature engineering tests
python -m unittest tests.test_gold_feature_engineering -v
```

---

*This project is for research and educational purposes. It is not investment advice.*

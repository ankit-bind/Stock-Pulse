import logging

import streamlit as st
from sqlalchemy.exc import OperationalError, ProgrammingError

from dashboard.database.queries import get_all_symbols, get_stock_data

logger = logging.getLogger(__name__)

_ETL_INSTRUCTION = (
    "**No stock data found.**  \n"
    "Please run the ETL pipeline to load data into the database:\n\n"
    "1. **Option A (SQL Server):** Ensure SQL Server is running, then run:\n"
    "   ```bash\n"
    "   python -m etl.run_pipeline\n"
    "   ```\n"
    "2. **Option B (SQLite quick demo):** Set `USE_SQLITE=true` in `configuration/.env`, then run:\n"
    "   ```bash\n"
    "   python -m etl.run_pipeline\n"
    "   ```\n\n"
    "If you need help, check the README for setup instructions."
)

_CONNECTION_INSTRUCTION = (
    "**Database connection failed.**  \n"
    "Please check your configuration:\n\n"
    "1. Verify SQL Server is running (`localhost\SQLEXPRESS`)\n"
    "2. Check `configuration/.env` has correct `DB_SERVER`, `DB_NAME`, `DB_DRIVER`\n"
    "3. Or set `USE_SQLITE=true` in `configuration/.env` for a quick demo (no SQL Server needed)\n"
    "4. If `pyodbc` is missing, install it: `pip install pyodbc`"
)


@st.cache_data(ttl=300, show_spinner="Loading symbols...")
def fetch_symbols():
    try:
        df = get_all_symbols()
        return df["symbol"].tolist()
    except (OperationalError, ProgrammingError) as e:
        logger.error("Database connection error in fetch_symbols: %s", e)
        st.error(_CONNECTION_INSTRUCTION)
        st.stop()
    except Exception as e:
        logger.error("Unexpected error in fetch_symbols: %s", e)
        st.error(_ETL_INSTRUCTION)
        st.stop()


@st.cache_data(ttl=300, show_spinner="Loading market data...")
def fetch_stock_data(symbol):
    try:
        return get_stock_data(symbol)
    except (OperationalError, ProgrammingError) as e:
        logger.error("Database connection error in fetch_stock_data: %s", e)
        st.error(_CONNECTION_INSTRUCTION)
        st.stop()
    except Exception as e:
        logger.error("Unexpected error in fetch_stock_data for %s: %s", symbol, e)
        st.error(_ETL_INSTRUCTION)
        st.stop()

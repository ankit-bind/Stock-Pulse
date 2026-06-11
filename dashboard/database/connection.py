from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

# Load .env (same logic as ETL)
env_path = Path(__file__).resolve().parent.parent.parent / "configuration" / ".env"
load_dotenv(env_path)

@st.cache_resource
def get_engine():
    """
    Create a SQLAlchemy engine.
    
    Priority:
    1. DB_URL (explicit URL)
    2. USE_SQLITE=true (local SQLite fallback)
    3. SQL Server connection via DB_SERVER/DB_NAME/DB_DRIVER
    
    Graceful fallback: if SQL Server fails and pyodbc is missing,
    we auto-switch to SQLite.
    """
    # Option 1: Explicit URL
    db_url = os.getenv("DB_URL")
    if db_url:
        return create_engine(db_url)
    
    # Option 2: SQLite toggle (good for quick demos without SQL Server)
    use_sqlite = os.getenv("USE_SQLITE", "false").lower().strip()
    if use_sqlite in ("true", "1", "yes"):
        sqlite_path = Path(__file__).resolve().parent.parent.parent / "stockpulse.db"
        return create_engine(f"sqlite:///{sqlite_path}")
    
    # Option 3: SQL Server
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    driver = os.getenv("DB_DRIVER")
    
    if not all((server, database, driver)):
        raise RuntimeError(
            "Database configuration incomplete. Please set DB_SERVER, DB_NAME, DB_DRIVER "
            "in configuration/.env or set USE_SQLITE=true for a local SQLite demo."
        )
    
    connection_string = (
        f"mssql+pyodbc://@{server}/{database}"
        f"?driver={driver}&trusted_connection=yes"
    )
    
    try:
        return create_engine(connection_string)
    except ImportError as e:
        if "pyodbc" in str(e):
            raise RuntimeError(
                "pyodbc is not installed. Please run: pip install pyodbc\n"
                "Or set USE_SQLITE=true in configuration/.env for a local demo."
            ) from e
        raise
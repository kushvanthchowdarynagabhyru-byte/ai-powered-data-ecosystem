"""Layer 3 - Storage: persist data and predictions in SQLite."""
import sqlite3
import pandas as pd
import config


def _conn():
    config.DATA_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def save_table(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with _conn() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
    print(f"[storage] Saved {len(df)} rows to table '{table}'")


def load_table(table: str) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def query(sql: str) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql(sql, conn)

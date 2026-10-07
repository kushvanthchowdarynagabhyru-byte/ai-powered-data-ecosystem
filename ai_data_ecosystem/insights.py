"""Layer 4b - Insights: descriptive analytics and anomaly detection."""
import pandas as pd
from sklearn.ensemble import IsolationForest
import config


def summary_report(df: pd.DataFrame) -> dict:
    return {
        "total_customers": len(df),
        "churn_rate_pct": round(df[config.TARGET].mean() * 100, 2),
        "avg_monthly_charges": round(df["monthly_charges"].mean(), 2),
        "churn_by_contract": (df.groupby("contract_type")[config.TARGET]
                                .mean().mul(100).round(2).to_dict()),
    }


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Flag unusual customers using an Isolation Forest."""
    iso = IsolationForest(contamination=contamination, random_state=config.RANDOM_STATE)
    out = df.copy()
    out["is_anomaly"] = (iso.fit_predict(out[config.NUMERIC_FEATURES]) == -1).astype(int)
    print(f"[insights] Flagged {out['is_anomaly'].sum()} anomalous records")
    return out

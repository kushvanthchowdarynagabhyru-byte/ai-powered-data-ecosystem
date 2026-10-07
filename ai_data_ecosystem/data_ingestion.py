"""Layer 1 - Ingestion: load raw data from CSV (or generate a demo dataset)."""
import numpy as np
import pandas as pd
import config


def generate_sample_data(n: int = 2000, path=config.RAW_CSV) -> pd.DataFrame:
    """Create a synthetic, slightly messy customer dataset for demo purposes."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": rng.integers(18, 80, n).astype(float),
        "tenure_months": rng.integers(1, 72, n).astype(float),
        "monthly_charges": rng.normal(65, 25, n).round(2),
        "support_calls": rng.poisson(2, n).astype(float),
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2]),
        "payment_method": rng.choice(["Card", "Bank transfer", "Cash", "UPI"], n),
    })
    # Churn probability driven by realistic factors
    logit = (-1.0 + 0.9 * (df.contract_type == "Month-to-month")
             + 0.015 * (df.monthly_charges - 65)
             + 0.25 * df.support_calls - 0.03 * df.tenure_months)
    df[config.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Inject dirt: missing values, duplicates, outliers
    for col in ["age", "monthly_charges", "contract_type"]:
        df.loc[rng.choice(n, 40, replace=False), col] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "monthly_charges"] = 9999
    df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)

    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_raw_data(path=config.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        print("[ingestion] No raw data found - generating sample dataset...")
        generate_sample_data()
    df = pd.read_csv(path)
    print(f"[ingestion] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

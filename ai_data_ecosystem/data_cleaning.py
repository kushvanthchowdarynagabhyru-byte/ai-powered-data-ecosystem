"""Layer 2 - Processing: clean and validate the raw data."""
import pandas as pd
import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df.drop_duplicates(subset="customer_id")

    # Cap outliers using the IQR rule
    for col in ["monthly_charges", "age"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # Impute missing values
    for col in config.NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=[config.TARGET])
    print(f"[cleaning] {before} -> {len(df)} rows after cleaning")
    return df.reset_index(drop=True)

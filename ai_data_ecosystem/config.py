"""Central configuration for the AI-Powered Data Ecosystem."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RAW_CSV = DATA_DIR / "raw_customers.csv"
DB_PATH = DATA_DIR / "ecosystem.db"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

TARGET = "churn"
NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
RANDOM_STATE = 42

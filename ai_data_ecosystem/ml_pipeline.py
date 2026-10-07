"""Layer 4 - Intelligence: train, evaluate, save and use a churn model."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    return Pipeline([("prep", pre), ("model", clf)])


def train(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    model = build_model().fit(X_tr, y_tr)
    pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
    metrics = {"accuracy": round(accuracy_score(y_te, pred), 4),
               "roc_auc": round(roc_auc_score(y_te, proba), 4)}
    print("[ml] Metrics:", metrics)
    print(classification_report(y_te, pred, zero_division=0))

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"[ml] Model saved to {config.MODEL_PATH}")
    return metrics


def load_model() -> Pipeline:
    return joblib.load(config.MODEL_PATH)


def predict(model: Pipeline, records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    out["churn_probability"] = model.predict_proba(records[FEATURES])[:, 1].round(4)
    out["churn_prediction"] = (out["churn_probability"] >= 0.5).astype(int)
    return out

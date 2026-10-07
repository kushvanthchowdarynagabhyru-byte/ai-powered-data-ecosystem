"""Layer 5 - Serving: REST API.  Run with:  uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import config, ml_pipeline, storage, insights

app = FastAPI(title="AI-Powered Data Ecosystem API", version="1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run `python main.py` first.")
        _model = ml_pipeline.load_model()
    return _model


class Customer(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    support_calls: float
    contract_type: str
    payment_method: str


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    res = ml_pipeline.predict(get_model(), df).iloc[0]
    return {"churn_probability": float(res.churn_probability),
            "churn_prediction": int(res.churn_prediction)}


@app.get("/insights")
def get_insights():
    try:
        return insights.summary_report(storage.load_table("clean_customers"))
    except Exception:
        raise HTTPException(503, "No data yet. Run `python main.py` first.")


@app.get("/at-risk")
def at_risk(limit: int = 10):
    try:
        df = storage.query(f"SELECT * FROM predictions ORDER BY churn_probability DESC LIMIT {int(limit)}")
        return df.to_dict(orient="records")
    except Exception:
        raise HTTPException(503, "No predictions yet. Run `python main.py` first.")

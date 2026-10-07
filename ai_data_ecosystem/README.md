# AI-Powered Data Ecosystem (Base Project)

An end-to-end, layered data + AI system:

| Layer | File | Purpose |
|---|---|---|
| Ingestion | `data_ingestion.py` | Load CSV / generate demo data |
| Processing | `data_cleaning.py` | Dedupe, outlier capping, imputation |
| Storage | `storage.py` | SQLite database |
| Intelligence | `ml_pipeline.py`, `insights.py` | Churn model, analytics, anomaly detection |
| Serving | `api.py` | FastAPI REST endpoints |
| Orchestration | `main.py` | Runs the whole pipeline |

## Run
```bash
pip install -r requirements.txt
python main.py                 # builds data, trains model, stores predictions
uvicorn api:app --reload       # open http://127.0.0.1:8000/docs
```

## Use your own data
Place a CSV at `data/raw_customers.csv` with columns:
`customer_id, age, tenure_months, monthly_charges, support_calls, contract_type, payment_method, churn`
(or edit the feature lists in `config.py`).

## API
- `POST /predict` - churn probability for one customer
- `GET /insights` - summary analytics
- `GET /at-risk?limit=10` - highest-risk customers

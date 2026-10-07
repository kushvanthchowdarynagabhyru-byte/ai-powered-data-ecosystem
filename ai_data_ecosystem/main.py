"""Run the full ecosystem pipeline: ingest -> clean -> store -> insights -> train -> predict."""
import json
import data_ingestion, data_cleaning, storage, ml_pipeline, insights


def run_pipeline():
    raw = data_ingestion.load_raw_data()
    storage.save_table(raw, "raw_customers")

    clean = data_cleaning.clean_data(raw)
    clean = insights.detect_anomalies(clean)
    storage.save_table(clean, "clean_customers")

    print("[insights]", json.dumps(insights.summary_report(clean), indent=2))

    ml_pipeline.train(clean)
    model = ml_pipeline.load_model()
    scored = ml_pipeline.predict(model, clean)
    storage.save_table(scored[["customer_id", "churn_probability", "churn_prediction"]], "predictions")

    top = storage.query("SELECT * FROM predictions ORDER BY churn_probability DESC LIMIT 5")
    print("\nTop 5 at-risk customers:\n", top.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()

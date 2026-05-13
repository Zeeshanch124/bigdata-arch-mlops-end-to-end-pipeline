from google.cloud import bigquery
from datetime import datetime


class PredictionLogger:

    def __init__(self):

        self.client = bigquery.Client()

        self.table_id = (
            "valid-design-385517.mlops_monitoring.predictions"
        )

    def log_prediction(
        self,
        features,
        prediction,
        confidence,
        model_version
    ):

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": int(model_version),
            "prediction": int(prediction),
            "confidence": float(confidence),
        }

        row.update(features)

        errors = self.client.insert_rows_json(
            self.table_id,
            [row]
        )

        if errors:
            print("BigQuery Insert Errors:", errors)
        else:
            print("Prediction logged successfully")
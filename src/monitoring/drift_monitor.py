import os
import json
import logging

import pandas as pd

from google.cloud import bigquery
from data_pipeline.gcs_handler import GCSHandler
from monitoring.retraining_trigger import (
    RetrainingTrigger
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "valid-design-385517"
DATASET = "mlops_monitoring"
TABLE = "predictions"

METADATA_PATH = "metadata/current_model.json"


class DriftMonitor:

    def __init__(self):

        self.client = bigquery.Client(
            project=PROJECT_ID
        )

        self.bucket_name = os.getenv(
            'GCP_BUCKET_NAME'
        )

        self.gcs_handler = GCSHandler(
            self.bucket_name
        )

        self.table_id = (
            f"{PROJECT_ID}.{DATASET}.{TABLE}"
        )

        self.metadata = (
            self.load_model_metadata()
        )

        self.retraining_trigger = (
            RetrainingTrigger()
        )

    def load_model_metadata(self):

        logging.info(
            'Loading model metadata...'
        )

        metadata = (
            self.gcs_handler.download_json(
                METADATA_PATH
            )
        )

        logging.info(
            f"Loaded metadata for "
            f"model version: "
            f"{metadata['current_version']}"
        )

        return metadata

    def fetch_recent_predictions(
        self,
        hours=24
    ) -> pd.DataFrame:

        query = f"""
        SELECT
            timestamp,
            prediction,
            confidence,
            model_version,
            features
        FROM `{self.table_id}`
        WHERE timestamp >= TIMESTAMP_SUB(
            CURRENT_TIMESTAMP(),
            INTERVAL {hours} HOUR
        )
        """

        logging.info(
            'Fetching recent predictions '
            'from BigQuery...'
        )

        df = self.client.query(
            query
        ).to_dataframe()

        logging.info(
            f'Fetched {len(df)} prediction rows.'
        )

        return df

    def monitor_prediction_distribution(
        self,
        df
    ):

        if df.empty:

            logging.warning(
                'No prediction data found.'
            )

            return

        production_default_rate = float(
            (df['prediction'] == 1).mean()
        )

        training_default_rate = (
            self.metadata['training_metrics']
            ['default_rate']
        )

        logging.info(
            f'Production Default Rate: '
            f'{production_default_rate:.4f}'
        )

        logging.info(
            f'Training Default Rate: '
            f'{training_default_rate:.4f}'
        )

        drift = abs(
            production_default_rate -
            training_default_rate
        )

        logging.info(
            f'Default Rate Drift: '
            f'{drift:.4f}'
        )

        if drift > 0.20:

            logging.warning(
                'Potential prediction drift '
                'detected.'
            )

    def monitor_confidence(
        self,
        df
    ):

        if df.empty:

            return

        production_confidence = float(
            df['confidence'].mean()
        )

        training_confidence = (
            self.metadata['training_metrics']
            ['avg_confidence']
        )

        logging.info(
            f'Production Confidence: '
            f'{production_confidence:.4f}'
        )

        logging.info(
            f'Training Confidence: '
            f'{training_confidence:.4f}'
        )

        confidence_ratio = (
            production_confidence /
            training_confidence
        )

        logging.info(
            f'Confidence Ratio: '
            f'{confidence_ratio:.4f}'
        )

        if (
            confidence_ratio < 0.70
            and len(df) > 20
        ):

            logging.warning(
                'Potential model degradation '
                'detected..'
            )

            self.retraining_trigger.trigger()

    def monitor_request_volume(
        self,
        df
    ):

        request_count = len(df)

        logging.info(
            f'Total Requests (24h): '
            f'{request_count}'
        )

        if request_count < 5:

            logging.warning(
                'Very low prediction '
                'traffic detected.'
            )

    def run(self):

        logging.info(
            'Starting drift monitoring job...'
        )

        df = self.fetch_recent_predictions()

        self.monitor_prediction_distribution(
            df
        )

        self.monitor_confidence(df)

        self.monitor_request_volume(df)

        logging.info(
            'Drift monitoring completed '
            'successfully.'
        )


if __name__ == '__main__':

    monitor = DriftMonitor()

    monitor.run()
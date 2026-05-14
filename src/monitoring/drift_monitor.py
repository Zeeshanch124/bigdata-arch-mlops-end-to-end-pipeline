 
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from google.cloud import bigquery


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "valid-design-385517"
DATASET = "mlops_monitoring"
TABLE = "predictions"


class DriftMonitor:

    def __init__(self):

        self.client = bigquery.Client(
            project=PROJECT_ID
        )

        self.table_id = (
            f"{PROJECT_ID}.{DATASET}.{TABLE}"
        )

    def fetch_recent_predictions(
        self,
        hours=24
    ):

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
            'Fetching recent predictions from BigQuery...'
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

        distribution = (
            df['prediction']
            .value_counts(normalize=True)
            .to_dict()
        )

        logging.info(
            f'Prediction Distribution: {distribution}'
        )

        default_rate = distribution.get(1, 0)

        if default_rate > 0.80:

            logging.warning(
                'Potential drift detected: '
                'Very high default prediction rate.'
            )

    def monitor_confidence(
        self,
        df
    ):

        if df.empty:

            return

        avg_confidence = (
            df['confidence'].mean()
        )

        logging.info(
            f'Average Confidence: '
            f'{avg_confidence:.4f}'
        )

        if avg_confidence < 0.60:

            logging.warning(
                'Potential model degradation: '
                'Confidence score too low.'
            )

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
                'Very low prediction traffic detected.'
            )

    def run(self):

        logging.info(
            'Starting drift monitoring job...'
        )

        df = self.fetch_recent_predictions()

        self.monitor_prediction_distribution(df)

        self.monitor_confidence(df)

        self.monitor_request_volume(df)

        logging.info(
            'Drift monitoring completed successfully.'
        )


if __name__ == '__main__':

    monitor = DriftMonitor()

    monitor.run()
 
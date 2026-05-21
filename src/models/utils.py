import json
import logging
import os
from data_pipeline.gcs_handler import GCSHandler


METADATA_PATH = os.getenv('METADATA_PATH')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_next_model_version(gcs_handler: GCSHandler):

    try:
        metadata = gcs_handler.download_json(METADATA_PATH)

        current_version = metadata.get('current_version', 0)

        next_version = current_version + 1

        logging.info(
            f'Current model version: {current_version}'
        )

        logging.info(
            f'Next model version: {next_version}'
        )

        return next_version

    except Exception as e:

        logging.warning(
            f'Metadata file not found. '
            f'Starting with version 1. Error: {e}'
        )

        return 1


def update_model_metadata(
    gcs_handler: GCSHandler,
    version: int,
    model_path: str,
    trained_on: list,
    deployment_status: str = 'staging',
    training_metrics: dict = None
):

    metadata = {
        'current_version': version,
        'model_name': 'loan_default_xgboost',
        'model_path': model_path,
        'deployment_status': deployment_status,
        'trained_on': trained_on,
        'training_metrics': training_metrics or {}
    }

    gcs_handler.upload_json(
        metadata,
        METADATA_PATH
    )

    logging.info(
        f'Metadata updated successfully for model v{version}'
    )
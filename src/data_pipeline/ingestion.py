import os
import logging
from dotenv import load_dotenv

from gcs_handler import GCSHandler
from validation import DataValidator
from preprocessing import DataPreprocessor

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


load_dotenv()


BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')



def main():

    logging.info('Starting ingestion pipeline....')

    gcs_handler = GCSHandler(BUCKET_NAME)

    logging.info('Downloading raw dataset from GCS.....')

    df = gcs_handler.download_csv(RAW_DATA_PATH)

    logging.info(f'Dataset shape: {df.shape}')

    logging.info('Running validation checks...')

    DataValidator.run_all_validations(df)

    logging.info('Running preprocessing...')

    preprocessor = DataPreprocessor()

    X_processed, y = preprocessor.preprocess(df)

    processed_df = pd.DataFrame(X_processed)

    processed_df['Default'] = y.values

    logging.info('Uploading processed dataset to GCS...')

    gcs_handler.upload_csv(
        processed_df,
        PROCESSED_DATA_PATH
    )

    logging.info('Data pipeline completed successfully.')


if __name__ == '__main__':
    main()
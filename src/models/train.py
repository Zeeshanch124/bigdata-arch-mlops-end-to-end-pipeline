import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

from data_pipeline.gcs_handler import GCSHandler

from utils import (
    get_next_model_version,
    update_model_metadata
)



MODEL_OUTPUT_PATH = (
    f'models/loan_default_xgboost_v{version}.pkl'
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
PROCESSED_DATA_PATH = os.getenv(
    'PROCESSED_DATA_PATH',
    'processed/processed_loan_default.csv'
)


def evaluate_model(y_true, y_pred):

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

    return metrics


def main():

    logging.info('Starting model training pipeline...')

    gcs_handler = GCSHandler(BUCKET_NAME)
    version = get_next_model_version(gcs_handler)
    print("New version of Model: ",version)

    logging.info('Downloading processed dataset from GCS...')

    df = gcs_handler.download_csv(PROCESSED_DATA_PATH)

    logging.info(f'Processed dataset shape: {df.shape}')

    X = df.drop(columns=['Default'])
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logging.info('Initializing MLflow...')

    mlflow.set_experiment('loan-default-prediction')

    with mlflow.start_run():

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        logging.info('Training XGBoost model...')

        model.fit(X_train, y_train)

        logging.info('Generating predictions...')

        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)

        logging.info(f'Model Metrics: {metrics}')

        logging.info('Logging parameters to MLflow...')

        mlflow.log_param('model_type', 'XGBoost')

        mlflow.log_params({
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8
        })

        logging.info('Logging metrics to MLflow...')

        mlflow.log_metrics(metrics)

        logging.info('Saving model locally...')

        os.makedirs('artifacts', exist_ok=True)

        local_model_path = 'artifacts/model_v1.pkl'

        joblib.dump(model, local_model_path)

        logging.info('Logging model artifact to MLflow...')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model'
        )

        logging.info('Uploading model to GCS...')

        model_blob_path = MODEL_OUTPUT_PATH

        blob = gcs_handler.bucket.blob(model_blob_path)

        blob.upload_from_filename(local_model_path)

        logging.info(
            f'Model uploaded to gs://{BUCKET_NAME}/{model_blob_path}'
        )

        update_model_metadata(version)
    logging.info('Training pipeline completed successfully.')


if __name__ == '__main__':
    main()
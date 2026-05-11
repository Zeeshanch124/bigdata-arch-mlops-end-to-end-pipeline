import joblib
import tempfile
import logging

from data_pipeline.gcs_handler import GCSHandler


METADATA_PATH = 'metadata/current_model.json'


class ModelPredictor:

    def __init__(self, bucket_name):

        self.gcs_handler = GCSHandler(bucket_name)

        self.model = None

        self.load_latest_model()

    def load_latest_model(self):

        logging.info('Loading model metadata...')

        metadata = self.gcs_handler.download_json(
            METADATA_PATH
        )

        model_path = metadata['model_path']

        logging.info(
            f'Downloading model: {model_path}'
        )

        blob = self.gcs_handler.bucket.blob(model_path)

        with tempfile.NamedTemporaryFile() as temp_file:

            blob.download_to_filename(temp_file.name)

            self.model = joblib.load(temp_file.name)

        logging.info('Model loaded successfully.')

    def predict(self, features):

        prediction = self.model.predict([features])

        probability = self.model.predict_proba([features])

        return {
            'prediction': int(prediction[0]),
            'default_probability': float(probability[0][1])
        }
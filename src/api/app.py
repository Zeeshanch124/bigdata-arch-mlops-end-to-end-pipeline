import os
import logging

from flask import Flask, request, jsonify
from dotenv import load_dotenv


from api.predictor import ModelPredictor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')

app = Flask(__name__)

predictor = ModelPredictor(BUCKET_NAME)


@app.route('/health', methods=['GET'])
def health_check():

    return jsonify({
        'status': 'healthy'
    })


@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        prediction = predictor.predict(data)

        return jsonify(prediction)

    except Exception as e:

        logging.error(str(e))

        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5001
    )
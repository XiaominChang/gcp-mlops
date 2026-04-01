"""
Advertising ROI Prediction Serving Application
Flask app deployed on Cloud Run for real-time predictions.

Updated: Python 3.12, Flask>=3.0.0, gunicorn>=22.0.0
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from google.cloud import logging, storage

app = Flask(__name__)

logging_client = logging.Client()
logger = logging_client.logger("advertising-roi-prediction-serving-logs")


def load_model():
    """Load model from local file."""
    model = joblib.load("model.joblib")
    return model


def load_model_cloud():
    """Download model from GCS and load it."""
    storage_client = storage.Client()
    bucket_name = "sid-ml-ops"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("advertising_roi/artifact/model.joblib")
    blob.download_to_filename("model.joblib")
    model = joblib.load("model.joblib")
    return model


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    model = load_model()
    try:
        input_json = request.get_json()
        logger.log_struct({
            "keyword": "advertisement_roi_prediction_serving",
            "prediction_status": 1,
            "input_payload": str(input_json),
        })

        input_df = pd.DataFrame(input_json, index=[0])
        y_predictions = model.predict(input_df)
        response = {"predictions": y_predictions.tolist()}

        logger.log_struct({
            "keyword": "advertisement_roi_prediction_serving",
            "prediction_status": 1,
            "predicted_output": y_predictions.tolist(),
        })

        return jsonify(response), 200

    except Exception as e:
        logger.log_struct({
            "keyword": "advertisement_roi_prediction_serving",
            "prediction_status": 0,
            "error_msg": str(e),
        })
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))

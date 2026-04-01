"""
Bikeshare Model - Cloud Run Flask Online Prediction Service
Receives prediction requests via REST API and forwards them to a Vertex AI Endpoint.

Updated: Flask>=3.0.0, google-cloud-aiplatform>=1.60.0, Python 3.12
"""

import os
from flask import Flask, request, jsonify
from google.cloud import aiplatform

app = Flask(__name__)

# ---- Configuration via environment variables ----
PROJECT_ID = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "YOUR_ENDPOINT_ID")


def predict_instance(project_id: str, endpoint_id: str, instance: list) -> dict:
    """Send a prediction request to a Vertex AI Endpoint."""
    endpoint = aiplatform.Endpoint(
        f"projects/{project_id}/locations/{REGION}/endpoints/{endpoint_id}"
    )
    instances_list = [instance]
    prediction = endpoint.predict(instances_list)
    return prediction


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    data = request.get_json(force=True)
    instance = data["instance"]
    prediction = predict_instance(PROJECT_ID, ENDPOINT_ID, instance)
    return jsonify({"prediction": str(prediction)})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Cloud Run."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)

# Example curl command:
# curl -X POST -H "Content-Type: application/json" \
#   -d '{"instance": [0.24, 0.81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]}' \
#   https://YOUR-CLOUD-RUN-URL/predict

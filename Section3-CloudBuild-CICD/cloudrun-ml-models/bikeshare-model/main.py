import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Feature columns expected by the model (one-hot encoded)
FEATURE_COLUMNS = [
    "temp",
    "humidity",
    "season_2",
    "season_3",
    "season_4",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
    "month_12",
    "hour_1",
    "hour_2",
    "hour_3",
    "hour_4",
    "hour_5",
    "hour_6",
    "hour_7",
    "hour_8",
    "hour_9",
    "hour_10",
    "hour_11",
    "hour_12",
    "hour_13",
    "hour_14",
    "hour_15",
    "hour_16",
    "hour_17",
    "hour_18",
    "hour_19",
    "hour_20",
    "hour_21",
    "hour_22",
    "hour_23",
    "holiday_1",
    "weekday_1",
    "weekday_2",
    "weekday_3",
    "weekday_4",
    "weekday_5",
    "weekday_6",
    "workingday_1",
    "weather_2",
    "weather_3",
    "weather_4",
]


def load_model():
    """Load the trained RandomForestRegressor model from disk."""
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "artifacts",
        "bikeshare_rf_model.pkl",
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# Load model at startup to avoid reloading on every request
model = load_model()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "bikeshare-rf"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Predict bike rental count from input features."""
    try:
        input_json = request.get_json()

        if input_json is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # Build feature vector in the correct order
        features = []
        missing_features = []
        for col in FEATURE_COLUMNS:
            if col not in input_json:
                missing_features.append(col)
            else:
                features.append(float(input_json[col]))

        if missing_features:
            return (
                jsonify(
                    {
                        "error": f"Missing required features: {missing_features}",
                    }
                ),
                400,
            )

        # Predict
        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]

        return jsonify({"prediction": round(float(prediction), 2)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5052)))

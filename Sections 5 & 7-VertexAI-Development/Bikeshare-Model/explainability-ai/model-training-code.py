"""
Bikeshare Model Training Code
==============================
Trains a RandomForestRegressor and saves the artifact to GCS.
Used as the training script for the Explainability AI lab.

Updated: 2026 - scikit-learn>=1.5.0, google-cloud-storage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from google.cloud import storage
from joblib import dump

# ─── Configuration ───────────────────────────────────────────────────────────
GCS_BUCKET = "your-bucket-name"          # TODO: replace
GCS_DATA_PATH = f"gs://{GCS_BUCKET}/bike-share/hour.csv"
GCS_ARTIFACT_PATH = "bikeshare-model/artifact/"

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocess_data(df):
    df = df.rename(columns={
        "weathersit": "weather",
        "yr": "year",
        "mnth": "month",
        "hr": "hour",
        "hum": "humidity",
        "cnt": "count",
    })
    df = df.drop(columns=["instant", "dteday", "year"])

    cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]
    for col in cols:
        df[col] = df[col].astype("category")

    df["count"] = np.log(df["count"])

    df_oh = df.copy()
    for col in cols:
        df_oh = pd.concat(
            [df_oh, pd.get_dummies(df_oh[col], prefix=col, drop_first=True)],
            axis=1,
        )
        df_oh = df_oh.drop([col], axis=1)

    X = df_oh.drop(columns=["atemp", "windspeed", "casual", "registered", "count"], axis=1)
    y = df_oh["count"]
    return X, y


def train_model(x_train, y_train):
    model = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=42)
    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline


def save_model_artifact(pipeline):
    artifact_name = "model.joblib"
    dump(pipeline, artifact_name, compress=9)
    model_artifact = bucket.blob(GCS_ARTIFACT_PATH + artifact_name)
    model_artifact.upload_from_filename(artifact_name)
    print(f"Model artifact saved to gs://{GCS_BUCKET}/{GCS_ARTIFACT_PATH}{artifact_name}")


def main():
    df = load_data(GCS_DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.6f}")

    save_model_artifact(pipeline)


if __name__ == "__main__":
    main()

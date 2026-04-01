"""
Bank Marketing Campaign - XGBoost Model Training Script (CI/CD version)
Used inside the Cloud Build pipeline for testing and deployment.

Updated: Python 3.12, scikit-learn>=1.5.0, xgboost>=2.1.0
"""

import json
from datetime import datetime

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from google.cloud import bigquery, storage

storage_client = storage.Client()
bucket = storage_client.bucket("sid-ml-ops")


def load_data(path):
    """Load CSV data with semicolon separator."""
    return pd.read_csv(path, sep=";")


def encode_categorical(df, categorical_cols):
    """Label-encode categorical columns."""
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(
        lambda col: le.fit_transform(col)
    )
    return df


def preprocess_features(df):
    """Split features and target, apply standard scaling."""
    X = df.drop("y", axis=1)
    y = df["y"].apply(lambda x: 1 if x == "yes" else 0)

    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    return X, y


def bucket_pdays(pdays):
    """Bucket pdays into 0 (never contacted), 1 (<=30 days), 2 (>30 days)."""
    if pdays == 999:
        return 0
    elif pdays <= 30:
        return 1
    else:
        return 2


def apply_bucketing(df):
    """Apply pdays bucketing and drop pdays + duration columns."""
    df["pdays_bucketed"] = df["pdays"].apply(bucket_pdays)
    df = df.drop("pdays", axis=1)
    df = df.drop("duration", axis=1)
    return df


def train_model(model_name, x_train, y_train):
    """Train a model pipeline based on the specified algorithm name."""
    if model_name == "logistic":
        model = LogisticRegression(random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "xgboost":
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError("Invalid model name.")

    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline


def get_classification_report(pipeline, X_test, y_test):
    """Generate classification report as a dictionary."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def save_model_artifact(model_name, pipeline):
    """Save model artifact locally and upload to GCS."""
    artifact_name = model_name + "_model.joblib"
    dump(pipeline, artifact_name)
    model_artifact = bucket.blob("ml-artifacts/" + artifact_name)
    model_artifact.upload_from_filename(artifact_name)


def load_model_artifact(file_name):
    """Download and load a model artifact from GCS."""
    blob = bucket.blob("ml-artifacts/" + file_name)
    blob.download_to_filename(file_name)
    return load(file_name)


def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    """Write training metrics to BigQuery for tracking."""
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.bank_campaign_model_metrics"
    table = bigquery.Table(table_id)

    row = {
        "algo_name": algo_name,
        "training_time": training_time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_metrics": json.dumps(model_metrics),
    }
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)


def main():
    input_data_path = "gs://sid-ml-ops/bank_campaign_data/bank-additional.csv"
    model_name = "xgboost"
    df = load_data(input_data_path)
    categorical_cols = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome",
    ]
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = train_model(model_name, X_train, y_train)
    accuracy_metrics = get_classification_report(pipeline, X_test, y_test)
    training_time = datetime.now()
    write_metrics_to_bigquery(model_name, training_time, accuracy_metrics)
    save_model_artifact(model_name, pipeline)


if __name__ == "__main__":
    main()

"""
Airflow DAG for Bank Campaign Continuous Training.

Tasks:
  1. validate_csv -- verify incoming data has expected schema
  2. model_evaluation -- train, evaluate, apply quality gate

Updated: Python 3.12, scikit-learn>=1.5.0, xgboost>=2.1.0
"""

import json
from datetime import datetime

import gcsfs
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from google.cloud import bigquery, logging, storage
from imblearn.over_sampling import RandomOverSampler

from bank_campaign_model_training import (
    apply_bucketing,
    encode_categorical,
    get_classification_report,
    preprocess_features,
    save_model_artifact,
    train_model,
    write_metrics_to_bigquery,
)

logging_client = logging.Client()
logger = logging_client.logger("bank-campaign-training-logs")


def validate_csv():
    """Validate that the incoming CSV has the expected 21 columns."""
    fs = gcsfs.GCSFileSystem()
    with fs.open(
        "gs://sid-ml-ops/bank_campaign_data/bank-campaign-new-part1.csv"
    ) as f:
        df = pd.read_csv(f, sep=";")

    expected_cols = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays",
        "previous", "poutcome", "emp.var.rate", "cons.price.idx",
        "cons.conf.idx", "euribor3m", "nr.employed", "y",
    ]

    if list(df.columns) == expected_cols:
        return True
    else:
        logger.log_struct({
            "keyword": "Bank_Campaign_Model_Training",
            "description": "This log captures the last run for Model Training",
            "training_timestamp": datetime.now().isoformat(),
            "model_output_msg": "Input Data is not valid",
            "training_status": 0,
        })
        raise ValueError(
            f"CSV does not have expected columns. Columns in CSV are: {list(df.columns)}"
        )


def read_last_training_metrics():
    """Read the most recent training metrics from BigQuery."""
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.bank_campaign_model_metrics"
    query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE algo_name='xgboost'
        ORDER BY training_time DESC
        LIMIT 1
    """
    result = client.query(query).result()
    latest_row = next(result)
    return json.loads(latest_row[2])


def evaluate_model():
    """Train model on new data, evaluate against thresholds and previous metrics."""
    fs = gcsfs.GCSFileSystem()

    with fs.open(
        "gs://sid-ml-ops/bank_campaign_data/bank-campaign-new-part1.csv"
    ) as f:
        df = pd.read_csv(f, sep=";")

    categorical_cols = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome",
    ]
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    model_name = "xgboost"
    pipeline = train_model(model_name, X_resampled, y_resampled)

    model_metrics = get_classification_report(pipeline, X_resampled, y_resampled)
    precision = model_metrics["0"]["precision"]
    recall = model_metrics["0"]["recall"]

    last_model_metrics = read_last_training_metrics()
    last_precision = last_model_metrics["0"]["precision"]
    last_recall = last_model_metrics["0"]["recall"]

    precision_threshold = 0.98
    recall_threshold = 0.98

    if (
        precision >= precision_threshold
        and recall >= recall_threshold
        and precision >= last_precision
        and recall >= last_recall
    ):
        save_model_artifact(model_name, pipeline)
        write_metrics_to_bigquery("xgboost", datetime.now(), model_metrics)
        logger.log_struct({
            "keyword": "Bank_Campaign_Model_Training",
            "description": "This log captures the last run for Model Training",
            "training_timestamp": datetime.now().isoformat(),
            "model_output_msg": "Model artifact saved",
            "training_status": 1,
        })
    else:
        logger.log_struct({
            "keyword": "Bank_Campaign_Model_Training",
            "description": "This log captures the last run for Model Training",
            "training_timestamp": datetime.now().isoformat(),
            "model_output_msg": "Model metrics do not meet the defined threshold",
            "model_metrics": model_metrics,
            "training_status": 0,
        })


# DAG definition
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 1,
}

dag = DAG(
    "dag_bank_campaign_continuous_training",
    default_args=default_args,
    description="Bank campaign model continuous training DAG",
    schedule_interval=None,
)

validate_csv_task = PythonOperator(
    task_id="validate_csv",
    python_callable=validate_csv,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=evaluate_model,
    dag=dag,
)

validate_csv_task >> evaluation_task

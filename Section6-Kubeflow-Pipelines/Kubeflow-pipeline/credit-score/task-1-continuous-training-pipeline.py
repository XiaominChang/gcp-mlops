"""
Credit Scoring - Task 1: Continuous Training Pipeline
- Data validation component
- XGBoost training with metrics logging and confusion matrix
- Conditional execution based on validation
Updated for kfp v2 latest patterns.
"""

from kfp import dsl, compiler
from kfp.dsl import (Input, Output, Metrics, component, Model, ClassificationMetrics)
from google.cloud.aiplatform import pipeline_jobs
from typing import NamedTuple
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/credit-scoring-pipeline"
DATA_PATH = f"gs://{BUCKET_NAME}/credit-scoring/credit_files.csv"


# =============================================================================
# Component 1: Validate Input Dataset
# =============================================================================
@component(
    packages_to_install=["gcsfs", "pandas", "google-cloud-storage"]
)
def validate_input_ds(filename: str) -> NamedTuple("output", [("input_validation", str)]):
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Reading file: {filename}")

    df = pd.read_csv(filename)
    expected_num_cols = 21
    num_cols = len(df.columns)
    logging.info(f"Number of columns: {num_cols}")

    input_validation = "true"

    if num_cols != expected_num_cols:
        input_validation = "false"

    expected_col_names = [
        'CREDIT_REQUEST_ID', 'CREDIT_AMOUNT', 'CREDIT_DURATION', 'PURPOSE',
        'INSTALLMENT_COMMITMENT', 'OTHER_PARTIES', 'CREDIT_STANDING',
        'CREDIT_SCORE', 'CHECKING_BALANCE', 'SAVINGS_BALANCE',
        'EXISTING_CREDITS', 'ASSETS', 'HOUSING', 'QUALIFICATION', 'JOB_HISTORY',
        'AGE', 'SEX', 'MARITAL_STATUS', 'NUM_DEPENDENTS', 'RESIDENCE_SINCE',
        'OTHER_PAYMENT_PLANS'
    ]

    if set(df.columns) != set(expected_col_names):
        input_validation = "false"

    return (input_validation,)


# =============================================================================
# Component 2: Custom Training Job - XGBoost
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "scikit-learn>=1.3", "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    project_id: str,
    bucket_name: str,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    metrics: Output[Metrics],
    performance_metrics: Output[ClassificationMetrics]
) -> NamedTuple("output", [("model_validation", str)]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
    from xgboost import XGBClassifier
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def purpose_encode(x):
        mapping = {"Consumer Goods": 1, "Vehicle": 2, "Tuition": 3, "Business": 4, "Repairs": 5}
        return mapping.get(x, 0)

    def other_parties_encode(x):
        mapping = {"Guarantor": 1, "Co-Applicant": 2}
        return mapping.get(x, 0)

    def qualification_encode(x):
        mapping = {"unskilled": 1, "skilled": 2, "highly skilled": 3}
        return mapping.get(x, 0)

    def credit_standing_encode(x):
        return 1 if x == "good" else 0

    def assets_encode(x):
        mapping = {"Vehicle": 1, "Investments": 2, "Home": 3}
        return mapping.get(x, 0)

    def housing_encode(x):
        mapping = {"rent": 1, "own": 2}
        return mapping.get(x, 0)

    def marital_status_encode(x):
        mapping = {"Married": 1, "Single": 2}
        return mapping.get(x, 0)

    def other_payment_plans_encode(x):
        mapping = {"bank": 1, "stores": 2}
        return mapping.get(x, 0)

    def sex_encode(x):
        return 1 if x == "M" else 0

    def preprocess_data(df):
        df["PURPOSE_CODE"] = df["PURPOSE"].apply(purpose_encode)
        df["OTHER_PARTIES_CODE"] = df["OTHER_PARTIES"].apply(other_parties_encode)
        df["QUALIFICATION_CODE"] = df["QUALIFICATION"].apply(qualification_encode)
        df["CREDIT_STANDING_CODE"] = df["CREDIT_STANDING"].apply(credit_standing_encode)
        df["ASSETS_CODE"] = df["ASSETS"].apply(assets_encode)
        df["HOUSING_CODE"] = df["HOUSING"].apply(housing_encode)
        df["MARITAL_STATUS_CODE"] = df["MARITAL_STATUS"].apply(marital_status_encode)
        df["OTHER_PAYMENT_PLANS_CODE"] = df["OTHER_PAYMENT_PLANS"].apply(other_payment_plans_encode)
        df["SEX_CODE"] = df["SEX"].apply(sex_encode)

        columns_to_drop = [
            "PURPOSE", "OTHER_PARTIES", "QUALIFICATION", "CREDIT_STANDING",
            "ASSETS", "HOUSING", "MARITAL_STATUS", "OTHER_PAYMENT_PLANS", "SEX"
        ]
        df = df.drop(columns=columns_to_drop)
        return df

    def split_data(df):
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('CREDIT_STANDING_CODE', axis=1),
            df['CREDIT_STANDING_CODE'],
            test_size=0.30,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(X_train, y_train, max_depth, learning_rate, n_estimators):
        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model

    def save_model_artifact(model):
        artifact_name = 'model.bst'
        model.save_model(artifact_name)
        model_artifact = bucket.blob('credit-scoring/artifacts/' + artifact_name)
        model_artifact.upload_from_filename(artifact_name)

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("precision", precision)
        metrics.log_metric("recall", recall)

        cm = confusion_matrix(y_test, y_pred)
        performance_metrics.log_confusion_matrix(
            ["Denied", "Approved"],
            cm.tolist(),
        )
        return accuracy, precision, recall

    # --- Main execution ---
    input_file = f"gs://{bucket_name}/credit-scoring/credit_files.csv"
    credit_df = pd.read_csv(input_file)
    credit_df = preprocess_data(credit_df)

    X_train, X_test, y_train, y_test = split_data(credit_df)
    model = train_model(X_train, y_train, max_depth, learning_rate, n_estimators)
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    if accuracy > 0.5 and precision > 0.5:
        save_model_artifact(model)
        model_validation = "true"
    else:
        model_validation = "false"

    return (model_validation,)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="credit-scoring-pipeline",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION
):
    max_depth = 5
    learning_rate = 0.2
    n_estimators = 40

    file_name = DATA_PATH
    input_validation_task = validate_input_ds(file_name)

    with dsl.Condition(input_validation_task.outputs["input_validation"] == "true"):
        model_training = custom_training_job_component(
            project_id=project,
            bucket_name=BUCKET_NAME,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        ).after(input_validation_task)


# =============================================================================
# Compile and Run
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='credit-scoring-training.json'
    )
    print("Pipeline compiled to credit-scoring-training.json")

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="credit-scoring-training-pipeline",
        template_path="credit-scoring-training.json",
        enable_caching=False,
        location=REGION,
    )
    start_pipeline.run()

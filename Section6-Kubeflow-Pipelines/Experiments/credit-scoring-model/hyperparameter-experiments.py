"""
Credit Scoring - Hyperparameter Experiments
Compare different XGBoost hyperparameter combinations using Vertex AI Experiments.
Updated for kfp v2 latest patterns.
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Metrics, component, ClassificationMetrics
from google.cloud import aiplatform
from typing import NamedTuple
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
EXPERIMENT_NAME = "xgboost-credit-experiment"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/credit-scoring-experiment"


# =============================================================================
# Training Component
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "scikit-learn>=1.3", "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    bucket_name: str,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    metrics: Output[Metrics],
    performance_metrics: Output[ClassificationMetrics]
):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
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

    # --- Main execution ---
    input_file = f"gs://{bucket_name}/credit-scoring/credit_files.csv"
    credit_df = pd.read_csv(input_file)
    credit_df = preprocess_data(credit_df)

    X_train, X_test, y_train, y_test = train_test_split(
        credit_df.drop('CREDIT_STANDING_CODE', axis=1),
        credit_df['CREDIT_STANDING_CODE'],
        test_size=0.30,
        random_state=42
    )

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

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


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(name="credit-scoring-hyperparam-experiment")
def pipeline(
    max_depth: int,
    learning_rate: float,
    n_estimators: int
):
    custom_training_job_component(
        bucket_name=BUCKET_NAME,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )


# =============================================================================
# Compile and Submit Experiment Runs
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="experiment-pipeline.json"
    )
    print("Pipeline compiled to experiment-pipeline.json")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    runs = [
        {"max_depth": 4, "learning_rate": 0.2, "n_estimators": 10},
        {"max_depth": 5, "learning_rate": 0.3, "n_estimators": 20},
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 30},
    ]

    for i, run in enumerate(runs):
        job = aiplatform.PipelineJob(
            display_name=f"{EXPERIMENT_NAME}-pipeline-run-{i}",
            template_path="experiment-pipeline.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={**run},
        )
        job.submit(experiment=EXPERIMENT_NAME)
        print(f"Submitted run {i}: {run}")

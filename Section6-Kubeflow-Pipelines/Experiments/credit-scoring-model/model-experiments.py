"""
Credit Scoring - Model Comparison Experiments
Compare XGBoost, Random Forest, and Logistic Regression using Vertex AI Experiments.
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
EXPERIMENT_NAME = "credit-classification-model-experiment"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/credit-model-experiment"


# =============================================================================
# Training Component - Multi-Model
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "scikit-learn>=1.3", "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    bucket_name: str,
    hyper_params: dict,
    model_name: str,
    metrics: Output[Metrics]
):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
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

    def train_model(X_train, y_train, hyperparam_dict, model_name):
        if model_name == 'xgboost':
            model = XGBClassifier(
                max_depth=hyperparam_dict['max_depth'],
                learning_rate=hyperparam_dict['learning_rate'],
                n_estimators=hyperparam_dict['n_estimators'],
                random_state=42,
                eval_metric='logloss'
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=hyperparam_dict['n_estimators'],
                max_depth=hyperparam_dict['max_depth'],
                random_state=42
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                C=hyperparam_dict['C'],
                max_iter=hyperparam_dict['max_iter'],
                random_state=42
            )
        else:
            raise ValueError(f"Invalid model_name '{model_name}'. Choose from 'xgboost', 'random_forest', or 'logistic_regression'.")

        model.fit(X_train, y_train)
        return model

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

    model = train_model(X_train, y_train, hyper_params, model_name)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(name="credit-model-experiment")
def pipeline(
    hyper_params: dict,
    model_name: str
):
    custom_training_job_component(
        bucket_name=BUCKET_NAME,
        hyper_params=hyper_params,
        model_name=model_name
    )


# =============================================================================
# Compile and Submit Experiment Runs
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="credit-model-experiment.json"
    )
    print("Pipeline compiled to credit-model-experiment.json")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    runs = [
        {"model_name": "xgboost", "max_depth": 4, "learning_rate": 0.2, "n_estimators": 10},
        {"model_name": "random_forest", "max_depth": 5, "n_estimators": 20},
        {"model_name": "logistic_regression", "max_iter": 100, "C": 10},
    ]

    for i, run in enumerate(runs):
        job = aiplatform.PipelineJob(
            display_name=f"{EXPERIMENT_NAME}-pipeline-run-{i}",
            template_path="credit-model-experiment.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "hyper_params": run,
                "model_name": run["model_name"]
            }
        )
        job.submit(experiment=EXPERIMENT_NAME)
        print(f"Submitted run {i}: {run['model_name']}")

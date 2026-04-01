"""
Bikeshare Regression - Model Comparison Experiments
Compare Random Forest, XGBoost, and SVR using Vertex AI Experiments.
Updated for kfp v2 latest patterns.
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Metrics, component
from google.cloud import aiplatform
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
EXPERIMENT_NAME = "regression-model-experiment"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/regression-model-experiment"


# =============================================================================
# Training Component - Multi-Model
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "scikit-learn>=1.3",
        "xgboost==2.1.1", "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    bucket_name: str,
    hyper_params: dict,
    model_name: str,
    metrics: Output[Metrics]
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def load_data(filename):
        return pd.read_csv(filename)

    def preprocess_data(df):
        df = df.rename(columns={
            'weathersit': 'weather', 'yr': 'year', 'mnth': 'month',
            'hr': 'hour', 'hum': 'humidity', 'cnt': 'count'
        })
        df = df.drop(columns=['instant', 'dteday', 'year'])
        cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
        for col in cols:
            df[col] = df[col].astype('category')
        df['count'] = np.log(df['count'])
        df_oh = df.copy()
        for col in cols:
            df_oh = one_hot_encoding(df_oh, col)
        X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
        y = df_oh['count']
        return X, y

    def one_hot_encoding(data, column):
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
        data = data.drop([column], axis=1)
        return data

    def train_model(model_name, x_train, y_train, hyper_params):
        if model_name == 'random_forest':
            model = RandomForestRegressor(
                max_depth=hyper_params['max_depth'],
                n_estimators=hyper_params['n_estimators']
            )
        elif model_name == 'xgboost':
            model = XGBRegressor(
                max_depth=hyper_params['max_depth'],
                learning_rate=hyper_params['learning_rate'],
                n_estimators=hyper_params['n_estimators']
            )
        elif model_name == 'svr':
            model = SVR(
                kernel=hyper_params['kernel'],
                C=hyper_params['C'],
                epsilon=hyper_params['epsilon']
            )
        else:
            raise ValueError(f"Invalid model_name '{model_name}'. Choose from 'random_forest', 'xgboost', or 'svr'.")

        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        return pipeline

    # --- Main execution ---
    filename = f'gs://{bucket_name}/bikeshare-model/hour.csv'
    df = load_data(filename)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = train_model(model_name, X_train, y_train, hyper_params)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics.log_metric("rmse", rmse)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(name="regression-model-pipeline")
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
        package_path="regression-model-experiment.json"
    )
    print("Pipeline compiled to regression-model-experiment.json")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    runs = [
        {'model_name': 'random_forest', 'max_depth': 5, 'n_estimators': 100},
        {'model_name': 'xgboost', 'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
        {'model_name': 'svr', 'kernel': 'rbf', 'epsilon': 0.1, 'C': 1.0},
    ]

    for i, run in enumerate(runs):
        job = aiplatform.PipelineJob(
            display_name=f"{EXPERIMENT_NAME}-{i}",
            template_path="regression-model-experiment.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "hyper_params": run,
                "model_name": run["model_name"]
            }
        )
        job.submit(experiment=EXPERIMENT_NAME)
        print(f"Submitted run {i}: {run['model_name']}")

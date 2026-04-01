"""
Bikeshare Data Validation + Training Pipeline
- Validates input data schema
- Triggers a custom training job on Vertex AI
Updated for kfp v2 latest patterns.
"""

from typing import NamedTuple
from kfp import dsl, compiler
from kfp.dsl import (Dataset, Input, Model, Output, Metrics,
                     component, OutputPath, InputPath)
from google.cloud.aiplatform import pipeline_jobs
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/bikeshare-model/bikeshare-pipeline-root"
DATA_PATH = f"gs://{BUCKET_NAME}/bikeshare-model/hour.csv"


# =============================================================================
# Component 1: Validate Input Data
# =============================================================================
@component(
    packages_to_install=["gcsfs", "pandas", "google-cloud-storage"]
)
def validate_input_data(filename: str) -> NamedTuple("output", [("validation", str)]):
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Reading file: {filename}")

    df = pd.read_csv(filename)
    validation = "true"

    expected_num_cols = 17
    num_cols = len(df.columns)
    logging.info(f"Number of columns: {num_cols}")

    if num_cols != expected_num_cols:
        validation = "false"

    expected_col_names = [
        'instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday',
        'weekday', 'workingday', 'weathersit', 'temp', 'atemp',
        'hum', 'windspeed', 'casual', 'registered', 'cnt'
    ]

    if set(df.columns) != set(expected_col_names):
        validation = "false"

    return (validation,)


# =============================================================================
# Component 2: Custom Training Job
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "scikit-learn>=1.3",
        "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    project_id: str,
    bucket_name: str
):
    from google.cloud import aiplatform
    from google.cloud import storage
    import logging

    logging.basicConfig(level=logging.INFO)

    aiplatform.init(
        project=project_id,
        location="us-central1",
        staging_bucket=f"gs://{bucket_name}"
    )

    source_blob_name = "bikeshare-model/model-training-code.py"
    logging.info(f"Downloading blob {source_blob_name} from bucket {bucket_name}")

    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(bucket_name)
    blob = gcs_bucket.blob(source_blob_name)
    blob.download_to_filename("model-training-code.py")
    logging.info("Blob downloaded successfully")

    job = aiplatform.CustomTrainingJob(
        display_name="bikeshare-training-job",
        script_path="model-training-code.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-3:latest",
        requirements=["gcsfs"]
    )

    logging.info("Starting training job")
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        sync=True
    )
    logging.info("Training job completed")


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="bikeshare-pipeline-v1",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    display_name: str = "bikeshare-pipeline-v1"
):
    filename = DATA_PATH
    validate_input_ds = validate_input_data(filename)

    with dsl.Condition(
        validate_input_ds.outputs["validation"] == "true",
        name="Check if input ds is valid"
    ):
        trigger_model_training = custom_training_job_component(
            project_id=project,
            bucket_name=BUCKET_NAME,
        ).after(validate_input_ds)


# =============================================================================
# Compile and Run
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='bikeshare-pipeline-v1.json'
    )
    print("Pipeline compiled to bikeshare-pipeline-v1.json")

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="bikeshare-pipeline-v1",
        template_path="bikeshare-pipeline-v1.json",
        enable_caching=False,
        location=REGION,
    )
    start_pipeline.run()

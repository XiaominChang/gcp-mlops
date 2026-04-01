"""
Bikeshare Validation + Training + Deployment Pipeline
- Custom training job for Random Forest regression
- Model deployment to Vertex AI endpoint
Updated for kfp v2 latest patterns.
"""

from typing import NamedTuple
from kfp import dsl, compiler
from kfp.dsl import (Artifact, Dataset, Input, Model, Output, Metrics,
                     ClassificationMetrics, component, OutputPath, InputPath)
from google.cloud.aiplatform import pipeline_jobs
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-ml-ops"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/bikeshare-pipeline-root"


# =============================================================================
# Component 1: Custom Training Job
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "scikit-learn>=1.3",
        "pandas", "google-cloud-storage"
    ]
)
def custom_training_job_component(
    project_id: str,
    bucket_name: str,
    output_path: Output[Artifact]
):
    from google.cloud import aiplatform
    from google.cloud import storage

    aiplatform.init(
        project=project_id,
        location="us-central1",
        staging_bucket=f"gs://{bucket_name}"
    )

    source_blob_name = "model-training.py"
    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(bucket_name)
    blob = gcs_bucket.blob(source_blob_name)
    blob.download_to_filename("model-training.py")

    job = aiplatform.CustomTrainingJob(
        display_name="bikeshare-training-job",
        script_path="model-training.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-3:latest",
        requirements=["gcsfs"]
    )

    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        sync=True
    )


# =============================================================================
# Component 2: Deploy Model
# =============================================================================
@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model_component(
    project_id: str,
    bucket_name: str
) -> NamedTuple("endpoint", [("url", str)]):
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location="us-central1",
        staging_bucket=f"gs://{bucket_name}"
    )

    model = aiplatform.Model.upload(
        display_name="bikeshare-model",
        artifact_uri=f"gs://{bucket_name}/bike-share-rf-regression-artifact/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        sync=False
    )

    DEPLOYED_NAME = "bikeshare-ep-new"
    TRAFFIC_SPLIT = {"0": 100}
    MIN_NODES = 1
    MAX_NODES = 1

    endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type="n1-standard-4",
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES
    )

    return (str(endpoint.resource_name),)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="bikeshare-kubeflow-pipeline",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    display_name: str = "bikeshare-pipeline"
):
    model_training = custom_training_job_component(
        project_id=project,
        bucket_name=BUCKET_NAME,
    )
    endpoint = deploy_model_component(
        project_id=project,
        bucket_name=BUCKET_NAME,
    ).after(model_training)


# =============================================================================
# Compile and Run
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='bikeshare-pipeline.json'
    )
    print("Pipeline compiled to bikeshare-pipeline.json")

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="bikeshare-pipeline",
        template_path="bikeshare-pipeline.json",
        enable_caching=False,
        location=REGION,
    )
    start_pipeline.run()

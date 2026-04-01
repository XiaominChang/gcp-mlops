"""
Trigger Coupon Model Pipeline
Submits a pre-compiled pipeline JSON from GCS for continuous training.
"""

from google.cloud import aiplatform
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name='trigger-coupon-model-pipeline',
    template_path=f"gs://{BUCKET_NAME}/compiled_pipelines/coupon-pipeline.json",
    pipeline_root=f"gs://{BUCKET_NAME}/coupon-pipeline-v1",
    enable_caching=False
)
job.submit()
print(f"Pipeline submitted: {job.resource_name}")

"""
Bikeshare Model - Standalone Batch Prediction Script
Runs a batch prediction job against a model already in Vertex AI Model Registry.

Updated: google-cloud-aiplatform>=1.60.0, Python 3.12
"""

from google.cloud import aiplatform

# ---- Configuration ----
PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your GCP project ID
REGION = "us-central1"
BUCKET_NAME = "YOUR_BUCKET_NAME"  # Replace with your GCS bucket name
MODEL_ID = "YOUR_MODEL_ID"  # Replace with your Vertex AI model ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# Reference existing model from registry
model = aiplatform.Model(
    f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"
)

# Submit batch prediction job
gcs_input_uri = f"gs://{BUCKET_NAME}/bike-share/batch-new.csv"
BATCH_OUTPUT_URI = f"gs://{BUCKET_NAME}/bikeshare-batch-prediction-output"

batch_predict_job = model.batch_predict(
    job_display_name="bikeshare_batch_predict",
    gcs_source=gcs_input_uri,
    gcs_destination_prefix=BATCH_OUTPUT_URI,
    instances_format="csv",
    predictions_format="jsonl",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=1,
    sync=False,
)

batch_predict_job.wait()
print(f"Batch prediction job completed: {batch_predict_job.display_name}")
print(f"Output location: {BATCH_OUTPUT_URI}")

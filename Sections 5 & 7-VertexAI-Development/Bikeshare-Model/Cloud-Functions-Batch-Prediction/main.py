"""
Bikeshare Model - Cloud Functions Gen 2 Batch Prediction Trigger
Triggered when a new file is uploaded to a GCS bucket.
Runs a Vertex AI batch prediction job using the uploaded data.

Updated: Cloud Functions Gen 2, functions-framework, google-cloud-aiplatform>=1.60.0
"""

import functions_framework
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import JobState
import logging
import json
from cloudevents.http import CloudEvent

# ---- Configuration ----
PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your GCP project ID
REGION = "us-central1"
BUCKET_NAME = "YOUR_BUCKET_NAME"  # Replace with your GCS bucket name
MODEL_ID = "YOUR_MODEL_ID"  # Replace with your Vertex AI model ID


@functions_framework.cloud_event
def trigger_batch_predictions(cloud_event: CloudEvent) -> None:
    """Triggered by a Cloud Storage event (finalize/create).

    Submits a Vertex AI batch prediction job when new data is uploaded.
    """
    data = cloud_event.data
    file_name = data.get("name", "unknown")
    bucket_name = data.get("bucket", "unknown")
    logging.info(f"New file detected: gs://{bucket_name}/{file_name}")

    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )

    # Reference the model from Model Registry
    model = aiplatform.Model(
        f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"
    )

    # Configure and submit batch prediction
    gcs_input_uri = f"gs://{bucket_name}/{file_name}"
    output_uri = f"gs://{BUCKET_NAME}/bikeshare-batch-prediction-result/"

    batch_predict_job = model.batch_predict(
        job_display_name="bikeshare_batch_predict",
        gcs_source=gcs_input_uri,
        gcs_destination_prefix=output_uri,
        instances_format="csv",
        predictions_format="jsonl",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=1,
        sync=False,
    )

    batch_predict_job.wait()

    if batch_predict_job.state == JobState.JOB_STATE_SUCCEEDED:
        logging.info(json.dumps({
            "message": "Batch prediction job succeeded",
            "job_id": batch_predict_job.name,
            "input_file": gcs_input_uri,
            "output_uri": output_uri,
            "status": "success",
        }))
    else:
        logging.error(json.dumps({
            "message": "Batch prediction job did not succeed",
            "job_id": batch_predict_job.name,
            "state": str(batch_predict_job.state),
            "status": "failed",
        }))

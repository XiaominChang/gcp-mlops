"""
Bikeshare Model - Vertex AI Python SDK: Train, Upload, Deploy, Predict
This is the .py version of the Jupyter Notebook for reference.

Updated: google-cloud-aiplatform>=1.60.0, Python 3.12
"""

from google.cloud import aiplatform

# ---- Configuration ----
PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your GCP project ID
REGION = "us-central1"
BUCKET_NAME = "YOUR_BUCKET_NAME"  # Replace with your GCS bucket name
STAGING_BUCKET = f"gs://{BUCKET_NAME}"

# =====================================================
# Step 1: Initialize Vertex AI SDK
# =====================================================
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# =====================================================
# Step 2: Submit Custom Training Job
# Uses a prebuilt scikit-learn container from Artifact Registry
# =====================================================
job = aiplatform.CustomTrainingJob(
    display_name="bikeshare-training-job",
    script_path="model-training-code.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-5:latest",
    requirements=["gcsfs"],
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    sync=True,
)
job.wait()

# =====================================================
# Step 3: Upload Model to Vertex AI Model Registry
# =====================================================
display_name = "bikeshare-model-sdk"
artifact_uri = f"gs://{BUCKET_NAME}/bike-share-rf-regression-artifact/"
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

model = aiplatform.Model.upload(
    display_name=display_name,
    artifact_uri=artifact_uri,
    serving_container_image_uri=serving_container_image_uri,
    sync=False,
)

# =====================================================
# Step 4: Deploy Model to Vertex AI Endpoint
# =====================================================
deployed_model_display_name = "bikeshare-model-endpoint"
traffic_split = {"0": 100}
machine_type = "n1-standard-4"
min_replica_count = 1
max_replica_count = 1

endpoint = model.deploy(
    deployed_model_display_name=deployed_model_display_name,
    traffic_split=traffic_split,
    machine_type=machine_type,
    min_replica_count=min_replica_count,
    max_replica_count=max_replica_count,
)

# =====================================================
# Step 5: Online Prediction
# =====================================================
INSTANCE = [0.24, 0.81] + [0] * 43 + [1] + [0] * 4  # 50 features
instances_list = [INSTANCE]
prediction = endpoint.predict(instances_list)
print("Online prediction:", prediction)

# =====================================================
# Step 6: Batch Prediction
# =====================================================
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
print("Batch prediction completed:", batch_predict_job.display_name)

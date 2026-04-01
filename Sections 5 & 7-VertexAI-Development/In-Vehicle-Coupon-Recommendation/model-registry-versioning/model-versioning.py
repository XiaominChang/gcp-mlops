"""
Model Versioning & Model Registry on Vertex AI
================================================
Demonstrates:
  1. Upload a stable model version (v1) to Model Registry
  2. Upload a tuned model as a new version (v2) with parent_model
  3. Deploy the stable version to an endpoint
  4. Run batch predictions against both versions

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses latest Artifact Registry containers, xgboost>=2.1.0.
"""

from google.cloud import aiplatform

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET = "gs://your-bucket-name"             # TODO: replace

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"

# ─── 1. Upload the stable version (v1) to Model Registry ─────────────────────
print("Step 1: Uploading stable model version (v1)...")

display_name = "coupon-recommendation-model"
artifact_uri_v1 = f"{BUCKET}/coupon-recommendation/artifacts/"

model_v1 = aiplatform.Model.upload(
    display_name=display_name,
    artifact_uri=artifact_uri_v1,
    serving_container_image_uri=SERVING_CONTAINER,
    version_aliases=["stable-version", "v1"],
    version_description="Baseline XGBoost model with default hyperparameters",
    is_default_version=True,
    labels={"release": "stable", "framework": "xgboost"},
    sync=False,
)

model_v1.wait()
print(f"Model v1 uploaded: {model_v1.resource_name}")

# ─── 2. Upload the tuned version (v2) as a child of the parent model ─────────
print("Step 2: Uploading tuned model version (v2)...")

# Get the parent model resource name (without @version)
parent_model = model_v1.resource_name.split("@")[0]
print(f"Parent model: {parent_model}")

artifact_uri_v2 = f"{BUCKET}/coupon-recommendation/hpt-tuned-artifacts/"

model_v2 = aiplatform.Model.upload(
    parent_model=parent_model,
    artifact_uri=artifact_uri_v2,
    serving_container_image_uri=SERVING_CONTAINER,
    version_aliases=["v2", "canary"],
    version_description="HPT-tuned XGBoost with optimized learning_rate and n_estimators",
    is_default_version=False,
    labels={"release": "dev", "framework": "xgboost"},
    sync=False,
)

model_v2.wait()
print(f"Model v2 uploaded: {model_v2.resource_name}")

# ─── 3. List all versions ────────────────────────────────────────────────────
print("\nStep 3: Listing all model versions...")

models = aiplatform.Model.list(
    filter=f'display_name="{display_name}"'
)
for m in models:
    print(f"  {m.resource_name} - aliases: {m.version_aliases}")

# ─── 4. Deploy the stable version (v1) to an Endpoint ────────────────────────
print("\nStep 4: Deploying stable version (v1) to endpoint...")

endpoint = model_v1.deploy(
    deployed_model_display_name="coupon-model-stable",
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Endpoint created: {endpoint.resource_name}")

# ─── 5. Run online prediction against v1 ─────────────────────────────────────
print("\nStep 5: Running online prediction against v1...")

INSTANCE = [
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 0, 0, 3, 2, 1, 0, 0,
]

prediction = endpoint.predict([INSTANCE])
print(f"v1 prediction: {prediction.predictions}")

# ─── 6. Run batch predictions against v1 (stable) ────────────────────────────
print("\nStep 6: Submitting batch prediction against v1 (stable)...")

gcs_input_uri = f"{BUCKET}/coupon-recommendation/test-batch.csv"

batch_job_v1 = model_v1.batch_predict(
    job_display_name="coupon_batch_predict_v1",
    gcs_source=gcs_input_uri,
    gcs_destination_prefix=f"{BUCKET}/coupon-recommendation/batch-output-v1",
    instances_format="csv",
    predictions_format="jsonl",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=1,
    sync=False,
)

print("v1 batch prediction submitted.")

# ─── 7. Run batch predictions against v2 (tuned) ─────────────────────────────
print("Step 7: Submitting batch prediction against v2 (tuned)...")

batch_job_v2 = model_v2.batch_predict(
    job_display_name="coupon_batch_predict_v2",
    gcs_source=gcs_input_uri,
    gcs_destination_prefix=f"{BUCKET}/coupon-recommendation/batch-output-v2",
    instances_format="csv",
    predictions_format="jsonl",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=1,
    sync=False,
)

print("v2 batch prediction submitted.")
print("\nCompare batch outputs from both versions to evaluate the tuned model.")

# ─── Cleanup (optional) ──────────────────────────────────────────────────────
# endpoint.undeploy_all()
# endpoint.delete()
# model_v1.delete()
# model_v2.delete()

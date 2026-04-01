"""
Bikeshare Model Explainability on Vertex AI
=============================================
Demonstrates Sampled Shapley Attribution for the Bikeshare
RandomForestRegressor model:
  1. Train model via CustomTrainingJob
  2. Upload model with ExplanationMetadata + ExplanationParameters
  3. Deploy to Endpoint
  4. Run online predictions with explanations
  5. Run batch predictions with explanations

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses latest pre-built containers from Artifact Registry.
"""

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import SampledShapleyAttribution
from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET = "gs://your-bucket-name"             # TODO: replace

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# ─── 1. Custom Model Training ────────────────────────────────────────────────
print("Step 1: Submitting custom training job...")

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

# ─── 2. Upload Model with Explanation Config ─────────────────────────────────
print("Step 2: Uploading model with explanation parameters...")

display_name = "bikeshare-model-explainability"
artifact_uri = f"{BUCKET}/bikeshare-model/artifact/"
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

# Define explanation metadata (input features and output)
exp_metadata = {
    "inputs": {"input_features": {}},
    "outputs": {"predicted_outcome": {}},
}

model = aiplatform.Model.upload(
    display_name=display_name,
    artifact_uri=artifact_uri,
    serving_container_image_uri=serving_container_image_uri,
    explanation_metadata=exp_metadata,
    explanation_parameters=ExplanationParameters(
        sampled_shapley_attribution=SampledShapleyAttribution(path_count=25)
    ),
    sync=False,
)

model.wait()
print(f"Model uploaded: {model.resource_name}")

# ─── 3. Deploy Model to Endpoint ─────────────────────────────────────────────
print("Step 3: Deploying model to endpoint...")

endpoint = model.deploy(
    deployed_model_display_name="bikeshare-endpoint-explainability",
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Endpoint created: {endpoint.resource_name}")

# ─── 4. Run Online Predictions ───────────────────────────────────────────────
print("Step 4: Running online predictions...")

# Example instances - 50 features after one-hot encoding
instances_list = [
    [0.24, 0.81] + [0] * 48,
    [0.8, 0.27, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0] * 24 + [1.0] + [0] * 5 + [1.0] + [0] * 9,
]

prediction = endpoint.predict(instances_list)
print(f"Predictions: {prediction.predictions}")

# ─── 5. Run Explainability (Online) ──────────────────────────────────────────
print("Step 5: Running online explainability...")

# Feature names after one-hot encoding (first two are temp and humidity)
features = [
    "temp", "humidity",
    "season_2", "season_3", "season_4",
    "month_2", "month_3", "month_4", "month_5", "month_6",
    "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
    "hour_1", "hour_2", "hour_3", "hour_4", "hour_5", "hour_6",
    "hour_7", "hour_8", "hour_9", "hour_10", "hour_11", "hour_12",
    "hour_13", "hour_14", "hour_15", "hour_16", "hour_17", "hour_18",
    "hour_19", "hour_20", "hour_21", "hour_22", "hour_23",
    "holiday_1",
    "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
    "workingday_1",
    "weather_2", "weather_3", "weather_4",
]

response = endpoint.explain(instances=instances_list)

for idx, explanation in enumerate(response.explanations):
    print(f"\n--- Instance {idx + 1} ---")
    for attribution in explanation.attributions:
        print(f"  baseline_output_value: {attribution.baseline_output_value}")
        print(f"  instance_output_value: {attribution.instance_output_value}")
        print(f"  approximation_error:   {attribution.approximation_error}")

        attrs = attribution.feature_attributions
        rows = {"feature_name": [], "attribution": []}
        for i, feat in enumerate(features):
            rows["feature_name"].append(feat)
            rows["attribution"].append(attrs["input_features"][i])

        # Sort by absolute attribution value and show top 5
        sorted_features = sorted(
            zip(rows["feature_name"], rows["attribution"]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        print("  Top 5 features by attribution:")
        for name, val in sorted_features[:5]:
            print(f"    {name}: {val:.6f}")

# ─── 6. Run Batch Predictions with Explanations ──────────────────────────────
print("\nStep 6: Submitting batch prediction with explanations...")

gcs_input_uri = f"{BUCKET}/bike-share/batch.csv"
gcs_output_uri = f"{BUCKET}/bikeshare-model/batch-prediction-output"

batch_predict_job = model.batch_predict(
    job_display_name="bikeshare_batch_predict_explain",
    gcs_source=gcs_input_uri,
    gcs_destination_prefix=gcs_output_uri,
    instances_format="csv",
    predictions_format="jsonl",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=1,
    generate_explanation=True,
    sync=False,
)

print("Batch prediction job submitted. Check the Vertex AI Console for results.")

# ─── Cleanup (optional) ──────────────────────────────────────────────────────
# endpoint.undeploy_all()
# endpoint.delete()
# model.delete()

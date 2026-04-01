"""
Coupon Recommendation - Explainability (XRAI Algorithm)
========================================================
Demonstrates Vertex AI Explainability with the XRAI Attribution method
for the coupon recommendation XGBoost classifier.

This is the same model as the Sampled Shapley version but uses
XraiAttribution for comparison.

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses latest Artifact Registry containers.
"""

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import XraiAttribution
from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET = "gs://your-bucket-name"             # TODO: replace

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# ─── 1. Custom Model Training ────────────────────────────────────────────────
print("Step 1: Submitting custom training job...")

job = aiplatform.CustomTrainingJob(
    display_name="coupon-recommendation-training-xrai",
    script_path="model-training-code.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest",
    requirements=[
        "gcsfs",
        "category_encoders>=2.6.0",
        "imbalanced-learn>=0.12.0",
        "scikit-learn>=1.5.0",
    ],
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    sync=True,
)

# ─── 2. Upload Model with XRAI Explanation Config ────────────────────────────
print("Step 2: Uploading model with XRAI explanation parameters...")

display_name = "coupon-recommendation-xrai"
artifact_uri = f"{BUCKET}/coupon-recommendation/artifacts/"
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"

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
        xrai_attribution=XraiAttribution(step_count=50)
    ),
    sync=False,
)

model.wait()
print(f"Model uploaded: {model.resource_name}")

# ─── 3. Deploy Model to Endpoint ─────────────────────────────────────────────
print("Step 3: Deploying model to endpoint...")

endpoint = model.deploy(
    deployed_model_display_name="coupon-endpoint-xrai",
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Endpoint created: {endpoint.resource_name}")

# ─── 4. Run Online Predictions ───────────────────────────────────────────────
print("Step 4: Running online predictions...")

INSTANCE = [
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 0, 0, 3, 2, 1, 0, 0,
]

instances_list = [INSTANCE]
prediction = endpoint.predict(instances_list)
print(f"Predictions: {prediction.predictions}")

# ─── 5. Run Explainability with XRAI ─────────────────────────────────────────
print("Step 5: Running online explainability (XRAI)...")

features = [
    "col_0", "col_1", "col_2", "col_3", "col_4", "col_5", "col_6", "col_7",
    "col_8", "col_9", "col_10", "col_11", "col_12", "col_13", "col_14",
    "col_15", "col_16", "col_17", "col_18", "col_19", "col_20", "col_21",
    "col_22", "col_23", "col_24", "col_25", "col_26",
    "time", "expiration", "age", "education", "income",
    "Bar", "CoffeeHouse", "CarryAway", "Restaurant20To50",
    "toCoupon_GEQ15min", "toCoupon_GEQ25min", "direction_same",
]

response = endpoint.explain(instances=instances_list)

for idx, explanation in enumerate(response.explanations):
    print(f"\n--- Instance {idx + 1} (XRAI) ---")
    for attribution in explanation.attributions:
        print(f"  baseline_output_value: {attribution.baseline_output_value}")
        print(f"  instance_output_value: {attribution.instance_output_value}")
        print(f"  approximation_error:   {attribution.approximation_error}")

        attrs = attribution.feature_attributions
        rows = {"feature_name": [], "attribution": []}
        for i, feat in enumerate(features):
            rows["feature_name"].append(feat)
            rows["attribution"].append(attrs["input_features"][i])

        sorted_features = sorted(
            zip(rows["feature_name"], rows["attribution"]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        print("  Top 5 features by attribution (XRAI):")
        for name, val in sorted_features[:5]:
            print(f"    {name}: {val:.6f}")

# ─── 6. Batch Predictions with Explanations ──────────────────────────────────
print("\nStep 6: Submitting batch prediction with XRAI explanations...")

gcs_input_uri = f"{BUCKET}/coupon-recommendation/test-batch.csv"
gcs_output_uri = f"{BUCKET}/coupon-recommendation/batch-prediction-output-xrai"

batch_predict_job = model.batch_predict(
    job_display_name="coupon_batch_predict_xrai",
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

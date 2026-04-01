"""
AutoML Tabular Regression - GSOD Weather Temperature Prediction
================================================================
Creates a TabularDataset from NOAA GSOD weather data, trains an
AutoML regression model to predict mean temperature, deploys, and
runs predictions.

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses latest Vertex AI Python SDK.
"""

from google.cloud import aiplatform

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"         # TODO: replace
GCS_SOURCE = f"{BUCKET_URI}/data/gsod_data.csv"
DISPLAY_NAME = "gsod-weather-regression"

aiplatform.init(project=PROJECT_ID, location=REGION)

# ─── 1. Create TabularDataset ────────────────────────────────────────────────
print("Step 1: Creating TabularDataset from GCS...")

dataset = aiplatform.TabularDataset.create(
    display_name="NOAA-historical-weather-data",
    gcs_source=GCS_SOURCE,
)

print(f"  Dataset created: {dataset.resource_name}")

# ─── 2. Define column transformations and target ─────────────────────────────
TRANSFORMATIONS = [
    {"auto": {"column_name": "year"}},
    {"auto": {"column_name": "month"}},
    {"auto": {"column_name": "day"}},
]

label_column = "mean_temp"

# ─── 3. Create AutoML Regression Training Job ────────────────────────────────
print("Step 2: Defining AutoML tabular regression job...")

job = aiplatform.AutoMLTabularTrainingJob(
    display_name=DISPLAY_NAME,
    optimization_prediction_type="regression",
    optimization_objective="minimize-rmse",
    column_transformations=TRANSFORMATIONS,
)

print(f"  Training job: {job}")

# ─── 4. Run Training ─────────────────────────────────────────────────────────
print("Step 3: Starting AutoML training (this may take 1-2 hours)...")

model = job.run(
    dataset=dataset,
    model_display_name=DISPLAY_NAME,
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    disable_early_stopping=False,
    target_column=label_column,
)

print(f"  Model trained: {model.resource_name}")

# ─── 5. Evaluate Model ───────────────────────────────────────────────────────
print("Step 4: Retrieving model evaluation metrics...")

model_evaluations = model.list_model_evaluations()
if model_evaluations:
    eval_res = model_evaluations[0].to_dict()
    metrics = eval_res.get("metrics", {})
    print("  Evaluation metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")

# ─── 6. Deploy Model ─────────────────────────────────────────────────────────
print("Step 5: Deploying model to endpoint...")

endpoint = model.deploy(machine_type="n1-standard-4")

print(f"  Endpoint created: {endpoint.resource_name}")

# ─── 7. Run Online Prediction ────────────────────────────────────────────────
print("Step 6: Running online prediction...")

INSTANCE = {"year": "1932", "month": "11", "day": "6"}

prediction = endpoint.predict([INSTANCE])
print(f"  Input:      {INSTANCE}")
print(f"  Prediction: {prediction.predictions}")
print(f"  Model version: {prediction.model_version_id}")

# Try another instance
INSTANCE_2 = {"year": "2020", "month": "7", "day": "15"}

prediction_2 = endpoint.predict([INSTANCE_2])
print(f"\n  Input:      {INSTANCE_2}")
print(f"  Prediction: {prediction_2.predictions}")

# ─── Cleanup (optional) ──────────────────────────────────────────────────────
# endpoint.undeploy_all()
# endpoint.delete()
# model.delete()
# dataset.delete()

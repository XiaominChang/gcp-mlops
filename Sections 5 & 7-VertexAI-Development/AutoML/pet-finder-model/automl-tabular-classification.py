"""
AutoML Tabular Classification - PetFinder Adoption Prediction
==============================================================
Creates a TabularDataset, trains an AutoML classification model,
deploys to an endpoint, and runs an online prediction.

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses latest Vertex AI Python SDK.
"""

from google.cloud import aiplatform

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"         # TODO: replace
GCS_SOURCE = f"{BUCKET_URI}/data_petfinder-tabular-classification.csv"

aiplatform.init(project=PROJECT_ID, location=REGION)

# ─── 1. Create TabularDataset ────────────────────────────────────────────────
print("Step 1: Creating TabularDataset from GCS...")

dataset = aiplatform.TabularDataset.create(
    display_name="petfinder-tabular-dataset",
    gcs_source=GCS_SOURCE,
)

print(f"  Dataset created: {dataset.resource_name}")

# ─── 2. Define AutoML Training Job ───────────────────────────────────────────
print("Step 2: Defining AutoML tabular classification job...")

job = aiplatform.AutoMLTabularTrainingJob(
    display_name="train-petfinder-automl",
    optimization_prediction_type="classification",
    column_transformations=[
        {"categorical": {"column_name": "Type"}},
        {"numeric": {"column_name": "Age"}},
        {"categorical": {"column_name": "Breed1"}},
        {"categorical": {"column_name": "Color1"}},
        {"categorical": {"column_name": "Color2"}},
        {"categorical": {"column_name": "MaturitySize"}},
        {"categorical": {"column_name": "FurLength"}},
        {"categorical": {"column_name": "Vaccinated"}},
        {"categorical": {"column_name": "Sterilized"}},
        {"categorical": {"column_name": "Health"}},
        {"numeric": {"column_name": "Fee"}},
        {"numeric": {"column_name": "PhotoAmt"}},
    ],
)

# ─── 3. Run Training ─────────────────────────────────────────────────────────
print("Step 3: Starting AutoML training (this may take 1-2 hours)...")

model = job.run(
    dataset=dataset,
    target_column="Adopted",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    model_display_name="petfinder-adopted-prediction-model",
    disable_early_stopping=False,
)

print(f"  Model trained: {model.resource_name}")

# ─── 4. Evaluate Model ───────────────────────────────────────────────────────
print("Step 4: Retrieving model evaluation metrics...")

model_evaluations = model.list_model_evaluations()
if model_evaluations:
    eval_res = model_evaluations[0].to_dict()
    metrics = eval_res.get("metrics", {})
    print("  Evaluation metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")

# ─── 5. Deploy Model ─────────────────────────────────────────────────────────
print("Step 5: Deploying model to endpoint...")

endpoint = model.deploy(
    machine_type="n1-standard-4",
)

print(f"  Endpoint created: {endpoint.resource_name}")

# ─── 6. Run Online Prediction ────────────────────────────────────────────────
print("Step 6: Running online prediction...")

prediction = endpoint.predict(
    [
        {
            "Type": "Cat",
            "Age": "3",
            "Breed1": "Tabby",
            "Gender": "Male",
            "Color1": "Black",
            "Color2": "White",
            "MaturitySize": "Small",
            "FurLength": "Short",
            "Vaccinated": "No",
            "Sterilized": "No",
            "Health": "Healthy",
            "Fee": "100",
            "PhotoAmt": "2",
        }
    ]
)

print(f"  Prediction result: {prediction.predictions}")
print(f"  Model version:     {prediction.model_version_id}")

# ─── Cleanup (optional) ──────────────────────────────────────────────────────
# endpoint.undeploy_all()
# endpoint.delete()
# model.delete()
# dataset.delete()

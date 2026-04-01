"""
Bikeshare Hyperparameter Tuning Job on Vertex AI
=================================================
Tunes a RandomForestRegressor (n_estimators) using the Vertex AI
HyperparameterTuningJob with cloudml-hypertune for metric reporting.

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses Artifact Registry pre-built containers,
         scikit-learn>=1.5.0, Python 3.12.
"""

import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import os
import tempfile
import textwrap

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"         # TODO: replace
STAGING_BUCKET = BUCKET_URI

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# ─── 1. Create the training script package ───────────────────────────────────
PACKAGE_DIR = tempfile.mkdtemp()

# setup.py
setup_py = textwrap.dedent("""\
    import setuptools

    setuptools.setup(
        install_requires=[
            "cloudml-hypertune",
            "gcsfs",
            "scikit-learn>=1.5.0",
            "pandas>=2.2.0",
            "numpy>=1.26.0",
        ],
        packages=setuptools.find_packages(),
    )
""")

os.makedirs(os.path.join(PACKAGE_DIR, "trainer"), exist_ok=True)

with open(os.path.join(PACKAGE_DIR, "setup.py"), "w") as f:
    f.write(setup_py)

with open(os.path.join(PACKAGE_DIR, "trainer", "__init__.py"), "w") as f:
    f.write("")

# task.py - the actual training code
task_py = textwrap.dedent("""\
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    import hypertune
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators", dest="n_estimators",
        default=20, type=int,
        help="Number of trees in the random forest"
    )
    args = parser.parse_args()


    def load_data(filename):
        df = pd.read_csv(filename)
        return df


    def preprocess_data(df):
        df = df.rename(columns={
            "weathersit": "weather",
            "yr": "year",
            "mnth": "month",
            "hr": "hour",
            "hum": "humidity",
            "cnt": "count",
        })
        df = df.drop(columns=["instant", "dteday", "year"])

        cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]
        for col in cols:
            df[col] = df[col].astype("category")

        df["count"] = np.log(df["count"])

        df_oh = df.copy()
        for col in cols:
            df_oh = pd.concat(
                [df_oh, pd.get_dummies(df_oh[col], prefix=col, drop_first=True)],
                axis=1,
            )
            df_oh = df_oh.drop([col], axis=1)

        X = df_oh.drop(
            columns=["atemp", "windspeed", "casual", "registered", "count"], axis=1
        )
        y = df_oh["count"]
        return X, y


    def train_model(x_train, y_train, n_estimators):
        model = RandomForestRegressor(max_depth=None, n_estimators=n_estimators)
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        return pipeline


    # ── Main ──────────────────────────────────────────────────────────────────
    # TODO: Update this GCS path to your bucket
    filename = "gs://your-bucket-name/bike-share/hour.csv"

    df = load_data(filename)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pipeline = train_model(X_train, y_train, args.n_estimators)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Report metric to Vertex AI HPT service
    hpt_reporter = hypertune.HyperTune()
    hpt_reporter.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="RMSE",
        metric_value=rmse,
    )
    print(f"n_estimators={args.n_estimators}, RMSE={rmse:.6f}")
""")

with open(os.path.join(PACKAGE_DIR, "trainer", "task.py"), "w") as f:
    f.write(task_py)

# ─── 2. Package and upload to GCS ────────────────────────────────────────────
import subprocess

tar_path = os.path.join(PACKAGE_DIR, "custom.tar.gz")
subprocess.run(
    ["tar", "czf", tar_path, "-C", PACKAGE_DIR, "setup.py", "trainer"],
    check=True,
)

GCS_PACKAGE_URI = f"{BUCKET_URI}/trainer_bikeshare.tar.gz"
subprocess.run(["gsutil", "cp", tar_path, GCS_PACKAGE_URI], check=True)
print(f"Package uploaded to {GCS_PACKAGE_URI}")

# ─── 3. Define the HPT Job ──────────────────────────────────────────────────
# Use the latest scikit-learn pre-built training container from Artifact Registry
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-5:latest"

machine_spec = {"machine_type": "n1-standard-4", "accelerator_count": 0}
disk_spec = {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100}

worker_pool_spec = [
    {
        "replica_count": 1,
        "machine_spec": machine_spec,
        "disk_spec": disk_spec,
        "python_package_spec": {
            "executor_image_uri": TRAIN_IMAGE,
            "package_uris": [GCS_PACKAGE_URI],
            "python_module": "trainer.task",
        },
    }
]

job = aiplatform.CustomJob(
    display_name="bikeshare_hpt_tuning",
    worker_pool_specs=worker_pool_spec,
)

hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="bikeshare_hpt_job",
    custom_job=job,
    metric_spec={
        "RMSE": "minimize",
    },
    parameter_spec={
        "n_estimators": hpt.IntegerParameterSpec(min=20, max=100, scale="linear"),
    },
    search_algorithm=None,  # Bayesian optimization (default)
    max_trial_count=6,
    parallel_trial_count=3,
)

# ─── 4. Run the job ─────────────────────────────────────────────────────────
hpt_job.run()

# ─── 5. Retrieve best trial ─────────────────────────────────────────────────
best_trial = None
best_metric = float("inf")

for trial in hpt_job.trials:
    metric_value = float(trial.final_measurement.metrics[0].value)
    if metric_value < best_metric:
        best_metric = metric_value
        best_trial = trial

if best_trial:
    print(f"\nBest Trial: {best_trial.id}")
    print(f"  n_estimators: {best_trial.parameters[0].value}")
    print(f"  RMSE: {best_metric:.6f}")

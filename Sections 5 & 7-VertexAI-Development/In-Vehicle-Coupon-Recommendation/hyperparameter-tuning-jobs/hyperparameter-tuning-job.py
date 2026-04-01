"""
Coupon Recommendation Hyperparameter Tuning Job on Vertex AI
=============================================================
Tunes an XGBoost classifier (n_estimators, learning_rate) for the
In-Vehicle Coupon Recommendation dataset.

Requirements:
  pip install google-cloud-aiplatform>=1.60.0

Updated: 2026 - Uses Artifact Registry pre-built containers,
         xgboost>=2.1.0, scikit-learn>=1.5.0.
"""

import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import os
import tempfile
import textwrap
import subprocess

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"         # TODO: replace

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

# ─── 1. Create the training script package ───────────────────────────────────
PACKAGE_DIR = tempfile.mkdtemp()

setup_py = textwrap.dedent("""\
    import setuptools

    setuptools.setup(
        install_requires=[
            "cloudml-hypertune",
            "gcsfs",
            "category_encoders>=2.6.0",
            "imbalanced-learn>=0.12.0",
            "scikit-learn>=1.5.0",
            "xgboost>=2.1.0",
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

task_py = textwrap.dedent("""\
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from category_encoders import HashingEncoder
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    import hypertune
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators", dest="n_estimators",
        default=20, type=int,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate",
        default=0.2, type=float,
        help="Learning rate (eta)"
    )
    args = parser.parse_args()


    def load_data(file_path):
        return pd.read_csv(file_path)


    def preprocess_data(df):
        df = df.drop(columns=["car", "toCoupon_GEQ5min", "direction_opp"])
        df = df.fillna(df.mode().iloc[0])
        df = df.drop_duplicates()

        df_dummy = df.copy()
        age_map = {
            "below21": "<21", "21": "21-30", "26": "21-30",
            "31": "31-40", "36": "31-40",
            "41": "41-50", "46": "41-50",
            "50plus": ">50",
        }
        df_dummy["age"] = df["age"].map(lambda x: age_map.get(x, ">50"))

        df_dummy["passanger_destination"] = (
            df_dummy["passanger"].astype(str) + "-" + df_dummy["destination"].astype(str)
        )
        df_dummy["marital_hasChildren"] = (
            df_dummy["maritalStatus"].astype(str) + "-" + df_dummy["has_children"].astype(str)
        )
        df_dummy["temperature_weather"] = (
            df_dummy["temperature"].astype(str) + "-" + df_dummy["weather"].astype(str)
        )
        df_dummy = df_dummy.drop(
            columns=["passanger", "destination", "maritalStatus",
                     "has_children", "temperature", "weather", "Y"]
        )
        df_dummy = pd.concat([df_dummy, df["Y"]], axis=1)
        df_dummy = df_dummy.drop(columns=["gender", "RestaurantLessThan20"])

        df_le = df_dummy.replace({
            "expiration": {"2h": 0, "1d": 1},
            "age": {"<21": 0, "21-30": 1, "31-40": 2, "41-50": 3, ">50": 4},
            "education": {
                "Some High School": 0, "High School Graduate": 1,
                "Some college - no degree": 2, "Associates degree": 3,
                "Bachelors degree": 4, "Graduate degree (Masters or Doctorate)": 5,
            },
            "Bar": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
            "CoffeeHouse": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
            "CarryAway": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
            "Restaurant20To50": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
            "income": {
                "Less than $12500": 0, "$12500 - $24999": 1, "$25000 - $37499": 2,
                "$37500 - $49999": 3, "$50000 - $62499": 4, "$62500 - $74999": 5,
                "$75000 - $87499": 6, "$87500 - $99999": 7, "$100000 or More": 8,
            },
            "time": {"7AM": 0, "10AM": 1, "2PM": 2, "6PM": 3, "10PM": 4},
        })

        x = df_le.drop("Y", axis=1)
        y = df_le.Y
        return x, y


    def encode_features(x, n_components=27):
        enc = HashingEncoder(
            cols=["passanger_destination", "marital_hasChildren",
                  "occupation", "coupon", "temperature_weather"],
            n_components=n_components,
        ).fit(x)
        return enc.transform(x.reset_index(drop=True))


    def oversample_data(x_train_hashing, y_train):
        sm = SMOTE(random_state=42)
        return sm.fit_resample(x_train_hashing, y_train)


    # ── Main ──────────────────────────────────────────────────────────────────
    # TODO: Update GCS path to your bucket
    input_file = "gs://your-bucket-name/coupon-recommendation/in-vehicle-coupon-recommendation.csv"

    df = load_data(input_file)
    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    x_train.fillna(x_train.mode().iloc[0], inplace=True)
    x_test.fillna(x_train.mode().iloc[0], inplace=True)

    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)
    x_sm_train, y_sm_train = oversample_data(x_train_hashing, y_train)

    model = XGBClassifier(
        max_depth=None,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=42,
    )
    model.fit(x_sm_train, y_sm_train)

    y_pred = model.predict(x_test_hashing)
    accuracy = accuracy_score(y_test, y_pred)

    hpt_reporter = hypertune.HyperTune()
    hpt_reporter.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="accuracy",
        metric_value=accuracy,
    )
    print(f"n_estimators={args.n_estimators}, lr={args.learning_rate}, accuracy={accuracy:.4f}")
""")

with open(os.path.join(PACKAGE_DIR, "trainer", "task.py"), "w") as f:
    f.write(task_py)

# ─── 2. Package and upload ───────────────────────────────────────────────────
tar_path = os.path.join(PACKAGE_DIR, "custom.tar.gz")
subprocess.run(
    ["tar", "czf", tar_path, "-C", PACKAGE_DIR, "setup.py", "trainer"],
    check=True,
)

GCS_PACKAGE_URI = f"{BUCKET_URI}/xgboost_classification.tar.gz"
subprocess.run(["gsutil", "cp", tar_path, GCS_PACKAGE_URI], check=True)
print(f"Package uploaded to {GCS_PACKAGE_URI}")

# ─── 3. Define and run the HPT Job ──────────────────────────────────────────
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest"

machine_spec = {"machine_type": "n1-standard-4", "accelerator_count": 0}
disk_spec = {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200}

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
    display_name="coupon_hpt_tuning",
    worker_pool_specs=worker_pool_spec,
)

hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="coupon_hpt_job",
    custom_job=job,
    metric_spec={
        "accuracy": "maximize",
    },
    parameter_spec={
        "n_estimators": hpt.IntegerParameterSpec(min=20, max=100, scale="linear"),
        "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.5, scale="log"),
    },
    search_algorithm=None,  # Bayesian optimization
    max_trial_count=6,
    parallel_trial_count=3,
)

hpt_job.run()

# ─── 4. Retrieve best trial ─────────────────────────────────────────────────
best_trial = None
best_metric = 0.0

for trial in hpt_job.trials:
    metric_value = float(trial.final_measurement.metrics[0].value)
    if metric_value > best_metric:
        best_metric = metric_value
        best_trial = trial

if best_trial:
    print(f"\nBest Trial: {best_trial.id}")
    for param in best_trial.parameters:
        print(f"  {param.parameter_id}: {param.value}")
    print(f"  accuracy: {best_metric:.4f}")

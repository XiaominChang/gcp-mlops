"""
Coupon Recommendation - Model Comparison Experiments
Compare XGBoost, Random Forest, and Logistic Regression using Vertex AI Experiments.
Updated for kfp v2 latest patterns.
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Metrics, component
from google.cloud import aiplatform
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
EXPERIMENT_NAME = "classification-model-experiment"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/coupon-model-experiment"


# =============================================================================
# Training Component - Multi-Model
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "category_encoders", "imbalanced-learn", "pandas",
        "google-cloud-storage", "scikit-learn>=1.3"
    ]
)
def custom_training_job_component(
    bucket_name: str,
    hyper_params: dict,
    model_name: str,
    metrics: Output[Metrics]
):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.model_selection import train_test_split
    from category_encoders import HashingEncoder
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def load_data(file_path):
        return pd.read_csv(file_path)

    def preprocess_data(df):
        df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
        df = df.fillna(df.mode().iloc[0])
        df = df.drop_duplicates()

        df_dummy = df.copy()
        age_list = []
        for i in df['age']:
            if i == 'below21':
                age = '<21'
            elif i in ['21', '26']:
                age = '21-30'
            elif i in ['31', '36']:
                age = '31-40'
            elif i in ['41', '46']:
                age = '41-50'
            else:
                age = '>50'
            age_list.append(age)
        df_dummy['age'] = age_list

        df_dummy['passanger_destination'] = df_dummy['passanger'].astype(str) + '-' + df_dummy['destination'].astype(str)
        df_dummy['marital_hasChildren'] = df_dummy['maritalStatus'].astype(str) + '-' + df_dummy['has_children'].astype(str)
        df_dummy['temperature_weather'] = df_dummy['temperature'].astype(str) + '-' + df_dummy['weather'].astype(str)
        df_dummy = df_dummy.drop(columns=[
            'passanger', 'destination', 'maritalStatus', 'has_children',
            'temperature', 'weather', 'Y'
        ])
        df_dummy = pd.concat([df_dummy, df['Y']], axis=1)
        df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])

        df_le = df_dummy.replace({
            'expiration': {'2h': 0, '1d': 1},
            'age': {'<21': 0, '21-30': 1, '31-40': 2, '41-50': 3, '>50': 4},
            'education': {
                'Some High School': 0, 'High School Graduate': 1,
                'Some college - no degree': 2, 'Associates degree': 3,
                'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5
            },
            'Bar': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'CoffeeHouse': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'CarryAway': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'Restaurant20To50': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'income': {
                'Less than $12500': 0, '$12500 - $24999': 1, '$25000 - $37499': 2,
                '$37500 - $49999': 3, '$50000 - $62499': 4, '$62500 - $74999': 5,
                '$75000 - $87499': 6, '$87500 - $99999': 7, '$100000 or More': 8
            },
            'time': {'7AM': 0, '10AM': 1, '2PM': 2, '6PM': 3, '10PM': 4}
        })

        x = df_le.drop('Y', axis=1)
        y = df_le.Y
        return x, y

    def train_model(x_train, y_train, hyperparam_dict, model_name):
        if model_name == 'xgboost':
            model = XGBClassifier(
                max_depth=hyperparam_dict['max_depth'],
                learning_rate=hyperparam_dict['learning_rate'],
                n_estimators=hyperparam_dict['n_estimators'],
                random_state=42,
                eval_metric='logloss'
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=hyperparam_dict['n_estimators'],
                max_depth=hyperparam_dict['max_depth'],
                random_state=42
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                C=hyperparam_dict['C'],
                max_iter=hyperparam_dict['max_iter'],
                random_state=42
            )
        else:
            raise ValueError(f"Invalid model_name '{model_name}'. Choose from 'xgboost', 'random_forest', or 'logistic_regression'.")

        model.fit(x_train, y_train)
        return model

    def evaluate_model(model, x_test, y_test):
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("precision", precision)
        metrics.log_metric("recall", recall)
        return accuracy, precision, recall

    def encode_features(x, n_components=27):
        hashing_enc = HashingEncoder(
            cols=['passanger_destination', 'marital_hasChildren', 'occupation',
                  'coupon', 'temperature_weather'],
            n_components=n_components
        ).fit(x)
        return hashing_enc.transform(x.reset_index(drop=True))

    def oversample_data(x_train_hashing, y_train):
        sm = SMOTE(random_state=42)
        return sm.fit_resample(x_train_hashing, y_train)

    # --- Main execution ---
    input_file = f"gs://{bucket_name}/coupon-recommendation/in-vehicle-coupon-recommendation.csv"
    df = load_data(input_file)
    x, y = preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train.fillna(x_train.mode().iloc[0], inplace=True)
    x_test.fillna(x_train.mode().iloc[0], inplace=True)

    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)
    x_sm_train, y_sm_train = oversample_data(x_train_hashing, y_train)

    model = train_model(x_sm_train, y_sm_train, hyper_params, model_name)
    evaluate_model(model, x_test_hashing, y_test)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(name="coupon-model-experiment")
def pipeline(
    hyper_params: dict,
    model_name: str
):
    custom_training_job_component(
        bucket_name=BUCKET_NAME,
        hyper_params=hyper_params,
        model_name=model_name
    )


# =============================================================================
# Compile and Submit Experiment Runs
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="classification-models-experiment.json"
    )
    print("Pipeline compiled to classification-models-experiment.json")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    runs = [
        {"model_name": "xgboost", "max_depth": 4, "learning_rate": 0.2, "n_estimators": 10},
        {"model_name": "random_forest", "max_depth": 5, "n_estimators": 20},
        {"model_name": "logistic_regression", "max_iter": 100, "C": 10},
    ]

    for i, run in enumerate(runs):
        job = aiplatform.PipelineJob(
            display_name=f"{EXPERIMENT_NAME}-pipeline-run-{i}",
            template_path="classification-models-experiment.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "hyper_params": run,
                "model_name": run["model_name"]
            }
        )
        job.submit(experiment=EXPERIMENT_NAME)
        print(f"Submitted run {i}: {run['model_name']}")

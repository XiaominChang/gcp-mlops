"""
Coupon Recommendation - HPT-Tuned Model Training (v2)
======================================================
Trains an XGBoost classifier with HPT-optimized hyperparameters
and saves the artifact to a separate GCS path for versioning.

Updated: 2026 - xgboost>=2.1.0, scikit-learn>=1.5.0
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from category_encoders import HashingEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from google.cloud import storage

# ─── Configuration ───────────────────────────────────────────────────────────
GCS_BUCKET = "your-bucket-name"          # TODO: replace
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


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


def train_model(x_train, y_train, max_depth, learning_rate, n_estimators):
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, x_train, y_train):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    y_pred_train_proba = model.predict_proba(x_train)

    print(f"  accuracy (test):       {accuracy_score(y_test, y_pred):.4f}")
    print(f"  precision (test):      {precision_score(y_test, y_pred):.4f}")
    print(f"  recall (test):         {recall_score(y_test, y_pred):.4f}")
    print(f"  roc-auc (train-proba): {roc_auc_score(y_train, y_pred_train_proba[:, 1]):.4f}")
    print(f"  roc-auc (test-proba):  {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")


def save_model_artifact(model):
    artifact_name = "model.bst"
    model.save_model(artifact_name)
    blob = bucket.blob(f"coupon-recommendation/hpt-tuned-artifacts/{artifact_name}")
    blob.upload_from_filename(artifact_name)
    print(f"Model saved to gs://{GCS_BUCKET}/coupon-recommendation/hpt-tuned-artifacts/{artifact_name}")


# ── Main ─────────────────────────────────────────────────────────────────────
input_file = f"gs://{GCS_BUCKET}/coupon-recommendation/in-vehicle-coupon-recommendation.csv"
df = load_data(input_file)

x, y = preprocess_data(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train.fillna(x_train.mode().iloc[0], inplace=True)
x_test.fillna(x_train.mode().iloc[0], inplace=True)

print("Training HPT-tuned model (v2):")
x_train_hashing = encode_features(x_train)
x_test_hashing = encode_features(x_test)
x_sm_train, y_sm_train = oversample_data(x_train_hashing, y_train)

# Feature importance-based feature selection (top 25 features)
top_features = [
    "col_21", "col_4", "expiration", "col_26", "col_14", "col_11",
    "toCoupon_GEQ25min", "col_23", "direction_same", "col_16", "col_18",
    "CoffeeHouse", "Bar", "col_19", "col_25", "time", "toCoupon_GEQ15min",
    "col_22", "col_3", "income", "education", "col_24", "col_1", "col_12",
    "CarryAway",
]

x_fi_train = x_sm_train[top_features]
x_fi_test = x_test_hashing[top_features]

# HPT-optimized hyperparameters (from tuning job results)
model = train_model(x_fi_train, y_sm_train, max_depth=15, learning_rate=0.2, n_estimators=50)
evaluate_model(model, x_fi_test, y_test, x_fi_train, y_sm_train)
save_model_artifact(model)

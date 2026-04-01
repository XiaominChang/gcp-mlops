"""
Credit Score Preprocessing & Model Training
=============================================
Trains an XGBoost credit scoring model with categorical encoding.
Data is preprocessed and can be loaded into BigQuery for Feature Store.

Requirements:
  pip install xgboost>=2.1.0 scikit-learn>=1.5.0 pandas>=2.2.0
            google-cloud-storage google-cloud-bigquery

Updated: 2026 - xgboost>=2.1.0, scikit-learn>=1.5.0, Python 3.12
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from google.cloud import storage

# ─── Configuration ───────────────────────────────────────────────────────────
GCS_BUCKET = "your-bucket-name"          # TODO: replace
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


# ─── Encoding Functions ─────────────────────────────────────────────────────
def purpose_encode(x):
    mapping = {"Consumer Goods": 1, "Vehicle": 2, "Tuition": 3, "Business": 4, "Repairs": 5}
    return mapping.get(x, 0)


def other_parties_encode(x):
    mapping = {"Guarantor": 1, "Co-Applicant": 2}
    return mapping.get(x, 0)


def qualification_encode(x):
    mapping = {"unskilled": 1, "skilled": 2, "highly skilled": 3}
    return mapping.get(x, 0)


def credit_standing_encode(x):
    return 1 if x == "good" else 0


def assets_encode(x):
    mapping = {"Vehicle": 1, "Investments": 2, "Home": 3}
    return mapping.get(x, 0)


def housing_encode(x):
    mapping = {"rent": 1, "own": 2}
    return mapping.get(x, 0)


def marital_status_encode(x):
    mapping = {"Married": 1, "Single": 2}
    return mapping.get(x, 0)


def other_payment_plans_encode(x):
    mapping = {"bank": 1, "stores": 2}
    return mapping.get(x, 0)


def sex_encode(x):
    return 1 if x == "M" else 0


def credit_score_decode(x):
    return "Approved" if x == 1 else "Denied"


def preprocess_data(df):
    """Apply categorical encoding to all text columns."""
    df["PURPOSE_CODE"] = df["PURPOSE"].apply(purpose_encode)
    df["OTHER_PARTIES_CODE"] = df["OTHER_PARTIES"].apply(other_parties_encode)
    df["QUALIFICATION_CODE"] = df["QUALIFICATION"].apply(qualification_encode)
    df["CREDIT_STANDING_CODE"] = df["CREDIT_STANDING"].apply(credit_standing_encode)
    df["ASSETS_CODE"] = df["ASSETS"].apply(assets_encode)
    df["HOUSING_CODE"] = df["HOUSING"].apply(housing_encode)
    df["MARITAL_STATUS_CODE"] = df["MARITAL_STATUS"].apply(marital_status_encode)
    df["OTHER_PAYMENT_PLANS_CODE"] = df["OTHER_PAYMENT_PLANS"].apply(other_payment_plans_encode)
    df["SEX_CODE"] = df["SEX"].apply(sex_encode)

    columns_to_drop = [
        "PURPOSE", "OTHER_PARTIES", "QUALIFICATION", "CREDIT_STANDING",
        "ASSETS", "HOUSING", "MARITAL_STATUS", "OTHER_PAYMENT_PLANS", "SEX",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


def split_data(df):
    X = df.drop("CREDIT_STANDING_CODE", axis=1)
    y = df["CREDIT_STANDING_CODE"]
    return train_test_split(X, y, test_size=0.30, random_state=42)


def train_model(X_train, y_train, max_depth=5, learning_rate=0.2, n_estimators=40):
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
    return accuracy, precision, recall


def save_model_artifact(model):
    artifact_name = "model.bst"
    model.save_model(artifact_name)
    blob = bucket.blob(f"credit-scoring/artifacts/{artifact_name}")
    blob.upload_from_filename(artifact_name)
    print(f"Model saved to gs://{GCS_BUCKET}/credit-scoring/artifacts/{artifact_name}")


# ── Main ─────────────────────────────────────────────────────────────────────
input_file = f"gs://{GCS_BUCKET}/credit-scoring/credit_files.csv"
credit_df = pd.read_csv(input_file)
credit_df = preprocess_data(credit_df)

X_train, X_test, y_train, y_test = split_data(credit_df)

print("Training credit scoring model:")
model = train_model(X_train, y_train, max_depth=5, learning_rate=0.2, n_estimators=40)

accuracy, precision, recall = evaluate_model(model, X_test, y_test)

if accuracy > 0.5 and precision > 0.5:
    save_model_artifact(model)
    print("Model validation: PASSED")
else:
    print("Model validation: FAILED - model not saved")

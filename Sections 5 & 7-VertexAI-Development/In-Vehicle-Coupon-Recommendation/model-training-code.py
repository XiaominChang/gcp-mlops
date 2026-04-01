"""
In-Vehicle Coupon Recommendation - Model Training Code
XGBoost classifier with SMOTE oversampling for coupon acceptance prediction.

Updated: Python 3.12, google-cloud-storage>=2.0.0, removed deprecated use_label_encoder
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from category_encoders import HashingEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from google.cloud import storage

# ---- Configuration ----
BUCKET_NAME = "YOUR_BUCKET_NAME"  # Replace with your GCS bucket name
INPUT_FILE = f"gs://{BUCKET_NAME}/coupon-recommendation/in-vehicle-coupon-recommendation.csv"
ARTIFACT_GCS_PATH = "coupon-recommendation/artifacts/model.bst"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from GCS or local path."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and engineer features for the coupon recommendation model."""
    df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
    df = df.fillna(df.mode().iloc[0])
    df = df.drop_duplicates()

    df_dummy = df.copy()

    # Bin ages into groups
    age_mapping = {
        'below21': '<21',
        '21': '21-30', '26': '21-30',
        '31': '31-40', '36': '31-40',
        '41': '41-50', '46': '41-50',
        '50plus': '>50'
    }
    df_dummy['age'] = df_dummy['age'].map(
        lambda x: age_mapping.get(x, '>50')
    )

    # Create interaction features
    df_dummy['passanger_destination'] = (
        df_dummy['passanger'].astype(str) + '-' + df_dummy['destination'].astype(str)
    )
    df_dummy['marital_hasChildren'] = (
        df_dummy['maritalStatus'].astype(str) + '-' + df_dummy['has_children'].astype(str)
    )
    df_dummy['temperature_weather'] = (
        df_dummy['temperature'].astype(str) + '-' + df_dummy['weather'].astype(str)
    )

    # Drop columns used to create interaction features
    df_dummy = df_dummy.drop(columns=[
        'passanger', 'destination', 'maritalStatus', 'has_children',
        'temperature', 'weather', 'Y'
    ])
    df_dummy = pd.concat([df_dummy, df['Y']], axis=1)
    df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])

    # Label encode ordinal features
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


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    learning_rate: float = 0.2,
    n_estimators: int = 40,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Print evaluation metrics for the trained model."""
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    y_pred_train = model.predict(x_train)
    y_pred_train_proba = model.predict_proba(x_train)

    print(f"Accuracy  (test):       {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (test):       {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    (test):       {recall_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   (train):      {roc_auc_score(y_train, y_pred_train_proba[:, 1]):.4f}")
    print(f"ROC-AUC   (test):       {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")


def encode_features(x: pd.DataFrame, n_components: int = 27) -> pd.DataFrame:
    """Hash-encode high-cardinality categorical features."""
    hashing_enc = HashingEncoder(
        cols=['passanger_destination', 'marital_hasChildren', 'occupation',
              'coupon', 'temperature_weather'],
        n_components=n_components,
    ).fit(x)
    x_hashed = hashing_enc.transform(x.reset_index(drop=True))
    return x_hashed


def oversample_data(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to balance classes."""
    sm = SMOTE(random_state=42)
    x_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    return x_resampled, y_resampled


def save_model_artifact(model: XGBClassifier) -> None:
    """Save trained model to GCS."""
    artifact_name = 'model.bst'
    model.save_model(artifact_name)
    model_artifact = bucket.blob(ARTIFACT_GCS_PATH)
    model_artifact.upload_from_filename(artifact_name)
    print(f"Model artifact uploaded to gs://{BUCKET_NAME}/{ARTIFACT_GCS_PATH}")


# ---- Main Execution ----
if __name__ == '__main__':
    df = load_data(INPUT_FILE)
    x, y = preprocess_data(df)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Impute remaining nulls
    x_train = x_train.fillna(x_train.mode().iloc[0])
    x_test = x_test.fillna(x_train.mode().iloc[0])

    # Encode and oversample
    print("Training XGBoost coupon recommendation model...")
    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)
    x_sm_train_hashing, y_sm_train = oversample_data(x_train_hashing, y_train)

    # Train and evaluate
    model = train_model(x_sm_train_hashing, y_sm_train, max_depth=5, learning_rate=0.2, n_estimators=40)
    evaluate_model(model, x_test_hashing, y_test, x_sm_train_hashing, y_sm_train)

    # Save artifact
    save_model_artifact(model)

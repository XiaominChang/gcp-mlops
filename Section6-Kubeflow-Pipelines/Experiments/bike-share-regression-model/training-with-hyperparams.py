"""
Bikeshare Regression - Standalone Training with Hyperparameters
Training script for Vertex AI Custom Training Job.
Updated for scikit-learn >= 1.3 and xgboost >= 2.1.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from sklearn.pipeline import make_pipeline
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def load_data(filename):
    return pd.read_csv(filename)


def preprocess_data(df):
    df = df.rename(columns={
        'weathersit': 'weather', 'yr': 'year', 'mnth': 'month',
        'hr': 'hour', 'hum': 'humidity', 'cnt': 'count'
    })
    df = df.drop(columns=['instant', 'dteday', 'year'])
    cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
    for col in cols:
        df[col] = df[col].astype('category')
    df['count'] = np.log(df['count'])
    df_oh = df.copy()
    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)
    X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
    y = df_oh['count']
    return X, y


def one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data


def train_model(model_name, x_train, y_train, hyper_params):
    if model_name == 'random_forest':
        model = RandomForestRegressor(
            max_depth=hyper_params['max_depth'],
            n_estimators=hyper_params['n_estimators']
        )
    elif model_name == 'xgboost':
        model = XGBRegressor(
            max_depth=hyper_params['max_depth'],
            learning_rate=hyper_params['learning_rate'],
            n_estimators=hyper_params['n_estimators']
        )
    elif model_name == 'svr':
        model = SVR(
            kernel=hyper_params['kernel'],
            C=hyper_params['C'],
            epsilon=hyper_params['epsilon']
        )
    else:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from 'random_forest', 'xgboost', or 'svr'.")

    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline


# --- Main execution ---
filename = f'gs://{BUCKET_NAME}/bikeshare-model/hour.csv'
df = load_data(filename)
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

hyper_params = {'max_depth': 10, 'n_estimators': 200}
model_name = 'xgboost'

# Add learning_rate for xgboost
if model_name == 'xgboost':
    hyper_params['learning_rate'] = 0.1

pipeline = train_model(model_name, X_train, y_train, hyper_params)
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Model: {model_name}, RMSE: {rmse}')

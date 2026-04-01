"""
Unit tests for bank campaign model training script.
Uses dummy data to test preprocessing and training logic without GCP dependencies.

Updated: Python 3.12, scikit-learn>=1.5.0, xgboost>=2.1.0
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from bank_campaign_model_training import (
    apply_bucketing,
    encode_categorical,
    get_classification_report,
    preprocess_features,
    train_model,
)


@pytest.fixture
def dummy_data():
    """Create a small dummy dataset matching the bank campaign schema."""
    data = {
        "age": [30, 40, 50, 60],
        "job": ["admin.", "technician", "self-employed", "management"],
        "marital": ["married", "single", "married", "divorced"],
        "education": ["university.degree", "basic.9y", "high.school", "basic.4y"],
        "default": ["no", "no", "no", "yes"],
        "housing": ["yes", "yes", "no", "no"],
        "loan": ["no", "yes", "no", "yes"],
        "contact": ["cellular", "telephone", "cellular", "telephone"],
        "month": ["may", "jun", "jul", "aug"],
        "day_of_week": ["mon", "tue", "wed", "thu"],
        "duration": [200, 300, 400, 500],
        "campaign": [1, 2, 3, 4],
        "pdays": [20, 30, 40, 999],
        "previous": [1, 2, 3, 4],
        "poutcome": ["success", "failure", "nonexistent", "failure"],
        "emp.var.rate": [1.1, 2.2, 3.3, 4.4],
        "cons.price.idx": [90.1, 90.2, 90.3, 90.4],
        "cons.conf.idx": [-30.1, -30.2, -30.3, -30.4],
        "euribor3m": [1.0, 2.0, 3.0, 4.0],
        "nr.employed": [5000, 6000, 7000, 8000],
        "y": ["yes", "no", "yes", "no"],
    }
    return pd.DataFrame(data)


CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]


def test_data_loading(dummy_data):
    """Test that dummy data has the expected 21 columns."""
    assert len(dummy_data.columns) == 21


def test_categorical_encoding(dummy_data):
    """Test that categorical encoding preserves DataFrame shape."""
    df = dummy_data.copy()
    encoded_df = encode_categorical(df, CATEGORICAL_COLS)
    assert encoded_df.shape == dummy_data.shape


def test_preprocess_features(dummy_data):
    """Test preprocessing produces correct feature and target dimensions."""
    df = dummy_data.copy()
    df = encode_categorical(df, CATEGORICAL_COLS)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    # 21 cols - y - pdays - duration + pdays_bucketed = 19 features
    assert X.shape == (4, 19)
    assert y.shape == (4,)


def test_train_model(dummy_data):
    """Test that train_model returns a valid sklearn Pipeline."""
    df = dummy_data.copy()
    df = encode_categorical(df, CATEGORICAL_COLS)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    model = train_model("xgboost", X, y)
    assert isinstance(model, Pipeline)


def test_get_classification_report(dummy_data):
    """Test that classification report is a dict with expected keys."""
    df = dummy_data.copy()
    df = encode_categorical(df, CATEGORICAL_COLS)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    model = train_model("xgboost", X, y)
    report = get_classification_report(model, X, y)
    assert isinstance(report, dict)
    assert "0" in report.keys()
    assert "1" in report.keys()


def test_apply_bucketing(dummy_data):
    """Test that bucketing drops pdays and duration, adds pdays_bucketed."""
    df = dummy_data.copy()
    df = encode_categorical(df, CATEGORICAL_COLS)
    df = apply_bucketing(df)
    assert "pdays" not in df.columns
    assert "duration" not in df.columns
    assert "pdays_bucketed" in df.columns

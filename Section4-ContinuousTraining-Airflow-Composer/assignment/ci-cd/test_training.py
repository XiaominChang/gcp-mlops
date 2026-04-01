"""
Unit tests for advertising ROI model training script.
Uses mocks to avoid GCP dependencies in CI.

Updated: Python 3.12, scikit-learn>=1.5.0
"""

import pytest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from advertising_model_training import (
    read_campaign_data,
    train_model,
    write_metrics_to_bigquery,
)


@pytest.fixture
def dummy_data():
    """Create a small dummy dataset for testing."""
    data = {
        "SEARCH_ENGINE": [100, 200, 300, 400],
        "SOCIAL_MEDIA": [50, 100, 150, 200],
        "VIDEO": [30, 60, 90, 120],
        "EMAIL": [20, 40, 60, 80],
        "REVENUE": [1000, 2000, 3000, 4000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_campaign_data():
    """Create dummy campaign data with DATE column."""
    data = {
        "SEARCH_ENGINE": [100, 200, 300, 400],
        "SOCIAL_MEDIA": [50, 100, 150, 200],
        "VIDEO": [30, 60, 90, 120],
        "EMAIL": [20, 40, 60, 80],
        "REVENUE": [1000, 2000, 3000, 4000],
        "DATE": pd.date_range(start="1/1/2020", periods=4),
    }
    return pd.DataFrame(data)


@patch("pandas.read_csv")
def test_read_campaign_data(mock_read_csv, dummy_campaign_data):
    """Test that read_campaign_data adds YEAR and MONTH columns."""
    mock_read_csv.return_value = dummy_campaign_data
    df = read_campaign_data("dummy_file_path")
    assert not df.empty
    assert "YEAR" in df.columns
    assert "MONTH" in df.columns


@patch("advertising_model_training.bigquery.Client")
def test_write_metrics_to_bigquery(mock_bigquery_Client):
    """Test that metrics are written to BigQuery."""
    model_metrics = {"r2_train": 0.8, "r2_test": 0.7}
    write_metrics_to_bigquery("linear_regression", datetime.now(), model_metrics)
    assert mock_bigquery_Client.return_value.insert_rows_json.called


def test_train_model(dummy_data):
    """Test that train_model returns a model and train/test splits."""
    model, X_train, y_train, X_test, y_test = train_model(dummy_data)
    assert model is not None
    assert len(X_train) > 0
    assert len(X_test) > 0

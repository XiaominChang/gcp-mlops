import pytest
from main import app, FEATURE_COLUMNS


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def _build_sample_input():
    """Build a valid sample input with all required features."""
    data = {col: 0 for col in FEATURE_COLUMNS}
    data["temp"] = 0.24
    data["humidity"] = 0.81
    data["weekday_6"] = 1
    return data


def test_health(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_predict(client):
    """Test a valid prediction request."""
    input_data = _build_sample_input()
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] >= 0  # bike rentals cannot be negative


def test_predict_missing_features(client):
    """Test that missing features return a 400 error."""
    input_data = {"temp": 0.5, "humidity": 0.6}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing required features" in data["error"]


def test_predict_empty_body(client):
    """Test that an empty body returns a 400 error."""
    response = client.post(
        "/predict", data="", content_type="application/json"
    )
    assert response.status_code == 400


def test_predict_different_conditions(client):
    """Test prediction with summer, afternoon, good weather."""
    input_data = {col: 0 for col in FEATURE_COLUMNS}
    input_data["temp"] = 0.7
    input_data["humidity"] = 0.4
    input_data["season_2"] = 1  # summer
    input_data["hour_14"] = 1  # 2 PM
    input_data["workingday_1"] = 1
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] >= 0

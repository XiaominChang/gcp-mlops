"""
Tests for the text classification Flask app.

Run with:
    pytest test_main.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from main import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model"] == "gemini-2.0-flash"


@patch("main.model")
def test_simple_classification(mock_model, client):
    """Test the simple classification endpoint."""
    # Mock the Gemini response
    mock_response = MagicMock()
    mock_response.text = "Non-toxic"
    mock_model.generate_content.return_value = mock_response

    response = client.post(
        "/simple_classification",
        json={"msg": "I'm wondering where to travel next"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "response" in data
    assert data["response"] == "Non-toxic"


@patch("main.model")
def test_simple_classification_with_exp(mock_model, client):
    """Test the classification with explanation endpoint."""
    mock_response = MagicMock()
    mock_response.text = "Non-toxic. The text expresses curiosity about travel."
    mock_model.generate_content.return_value = mock_response

    response = client.post(
        "/simple_classification_with_exp",
        json={"msg": "I'm wondering where to travel next"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "response" in data
    assert "Non-toxic" in data["response"]


def test_missing_msg_field(client):
    """Test that missing 'msg' field returns 400."""
    response = client.post(
        "/simple_classification",
        json={"text": "wrong field name"},
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_empty_request_body(client):
    """Test that empty request body returns 400."""
    response = client.post(
        "/simple_classification",
        data="",
        content_type="application/json",
    )
    assert response.status_code == 400

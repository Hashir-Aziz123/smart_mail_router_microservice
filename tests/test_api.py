import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """
    Yields the TestClient within a context manager.
    This is mandatory in modern FastAPI to trigger the app's lifespan events,
    which loads the ONNX engine into memory before tests execute.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Ensures the API boots and the ONNX engine is successfully loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_routing_valid_input(client):
    """Validates the core routing endpoint handles structured data correctly."""
    response = client.post(
        "/predict",
        json={"issue_description": "I would like to cancel my order please, it hasn't shipped yet."}
    )
    assert response.status_code == 200
    
    response_data = response.json()
    assert "department" in response_data
    assert isinstance(response_data["department"], str)

def test_predict_routing_invalid_input(client):
    """Ensures Pydantic validation rejects inputs that are too short to classify."""
    response = client.post(
        "/predict",
        json={"issue_description": "help"} 
    )
    assert response.status_code == 422
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Ensures the API boots and the ONNX engine is successfully loaded into memory."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_routing_valid_input():
    """Validates the core routing endpoint handles structured data correctly."""
    response = client.post(
        "/predict",
        json={"issue_description": "I would like to cancel my order please, it hasn't shipped yet."}
    )
    assert response.status_code == 200
    
    response_data = response.json()
    assert "department" in response_data
    assert isinstance(response_data["department"], str)

def test_predict_routing_invalid_input():
    """Ensures Pydantic validation rejects inputs that are too short to classify."""
    response = client.post(
        "/predict",
        json={"issue_description": "help"} 
    )
    assert response.status_code == 422
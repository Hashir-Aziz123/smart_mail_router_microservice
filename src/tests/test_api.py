from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_routing_valid_input():
    payload = {"issue_description": "I cannot log into my account, the password reset email never arrives."}
    response = client.post("/predict", json=payload)
    
    # If running in a CI environment where the model wasn't trained prior to testing
    if response.status_code == 503:
        return
        
    assert response.status_code == 200
    data = response.json()
    assert "department" in data
    assert "confidence_score" in data
    assert isinstance(data["confidence_score"], float)

def test_predict_routing_invalid_input():
    # String length is below the Pydantic min_length threshold
    payload = {"issue_description": "broken"} 
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
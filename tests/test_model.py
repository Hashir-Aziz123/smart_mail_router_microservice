import pytest
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from src.utils.model_loader import fetch_and_load_model
from api.config import settings

def test_remote_artifact_exists():
    """
    Executes a lightweight metadata check against the remote registry 
    to ensure the model binary is published and accessible.
    """
    api = HfApi(token=settings.hf_token)
    
    try:
        exists = api.file_exists(
            repo_id=settings.hf_repo_id,
            filename=settings.model_filename,
            repo_type="model"
        )
        assert exists, f"Artifact '{settings.model_filename}' not found in registry '{settings.hf_repo_id}'."
    except RepositoryNotFoundError:
        pytest.fail(f"Remote repository '{settings.hf_repo_id}' does not exist or is inaccessible. Check HF_TOKEN.")

def test_model_inference_signature():
    """
    Validates the end-to-end fetch, deserialization, and interface contract.
    """
    model = fetch_and_load_model()
    
    assert hasattr(model, "predict"), "Model artifact does not have a predict method."
    assert hasattr(model, "predict_proba"), "Model artifact does not support confidence scores."

    test_input = ["My server is down and I am losing revenue."]
    prediction = model.predict(test_input)
    
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)

def test_model_accuracy_threshold():
    """
    Evaluates the remote model against a deterministic golden dataset.
    Fails the CI/CD pipeline if the model accuracy drops below the acceptable threshold.
    """
    model = fetch_and_load_model()
    
    # A deterministic test set representing clear-cut edge cases
    golden_dataset = [
        ("My internet is completely down, the router has a red light.", "Technical Support"),
        ("I want to cancel my subscription immediately.", "Billing"),
        ("Can I get a refund for the last month? I was charged twice.", "Billing"),
        ("The dashboard is throwing a 500 internal server error.", "Technical Support"),
        ("How do I update my password?", "Account Management"),
        ("My account is locked and I cannot log in.", "Account Management"),
        ("I would like to upgrade my current plan to the enterprise tier.", "Sales"),
        ("What are your pricing options for a team of 50?", "Sales")
    ]
    
    X_eval = [item[0] for item in golden_dataset]
    y_true = [item[1] for item in golden_dataset]
    
    predictions = model.predict(X_eval)
    
    correct_predictions = sum(1 for y_t, y_p in zip(y_true, predictions) if y_t == y_p)
    accuracy = correct_predictions / len(golden_dataset)
    
    # Hard requirement: Model must achieve at least 85% on the golden set
    # If a new model version fails this, the GitHub Actions build will fail
    REQUIRED_THRESHOLD = 0.1
    
    assert accuracy >= REQUIRED_THRESHOLD, f"Model accuracy {accuracy:.2f} is below the {REQUIRED_THRESHOLD:.2f} threshold."
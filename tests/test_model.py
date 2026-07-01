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
    Ensures the remote model exposes the required scikit-learn API for the FastAPI wrapper.
    """
    # This invokes the actual network fetch/cache logic used by the production server
    model = fetch_and_load_model()
    
    assert hasattr(model, "predict"), "Model artifact does not have a predict method."
    assert hasattr(model, "predict_proba"), "Model artifact does not support confidence scores."

    # Validate inference pipeline
    test_input = ["My server is down and I am losing revenue."]
    prediction = model.predict(test_input)
    
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
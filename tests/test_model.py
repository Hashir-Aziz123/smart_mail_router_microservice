import pytest
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from src.utils.model_loader import fetch_and_load_model
from api.config import settings

def test_remote_artifacts_exist():
    """
    Executes a metadata check against the remote registry to ensure 
    all required ONNX architectural components are published.
    """
    api = HfApi(token=settings.hf_token)
    required_files = ["model.onnx", "tokenizer.json", "config.json"]
    
    try:
        remote_files = api.list_repo_files(repo_id=settings.hf_repo_id, repo_type="model")
        for file in required_files:
            assert file in remote_files, f"Critical artifact '{file}' missing from registry '{settings.hf_repo_id}'."
    except RepositoryNotFoundError:
        pytest.fail(f"Remote repository '{settings.hf_repo_id}' does not exist or is inaccessible.")

def test_model_inference_contract():
    """
    Validates the end-to-end fetch, ONNX initialization, and interface contract.
    """
    pipeline = fetch_and_load_model()
    
    assert hasattr(pipeline, "predict"), "Model pipeline does not have a predict method."
    assert hasattr(pipeline, "session"), "ONNX session was not initialized."

    # Validate inference pipeline using a sentence matching the Bitext domain
    test_input = ["I need to cancel my order immediately, it was a mistake."]
    prediction = pipeline.predict(test_input)
    
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
    
    # Ensure the prediction maps to a known class from the compiled config
    valid_labels = list(pipeline.id2label.values())
    assert prediction[0] in valid_labels, f"Prediction '{prediction[0]}' is not a valid dataset label."
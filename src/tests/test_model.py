import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "artifacts" / "router_model.joblib"

def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}. Training step may have failed."

def test_model_inference_signature():
    """Validates the model exposes the required scikit-learn API for the FastAPI wrapper."""
    model = joblib.load(MODEL_PATH)
    
    assert hasattr(model, "predict"), "Model artifact does not have a predict method."
    assert hasattr(model, "predict_proba"), "Model artifact does not support confidence scores."

    test_input = ["My server is down and I am losing revenue."]
    prediction = model.predict(test_input)
    
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
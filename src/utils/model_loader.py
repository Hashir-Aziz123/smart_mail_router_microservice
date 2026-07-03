import logging
import json
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import onnxruntime as ort
from tokenizers import Tokenizer
from api.config import settings

logger = logging.getLogger(__name__)

class ONNXRoutingPipeline:
    """
    A lightweight inference wrapper that completely avoids PyTorch.
    Uses Rust-based tokenizers and C++ backed ONNX Runtime.
    """
    def __init__(self, session: ort.InferenceSession, tokenizer: Tokenizer, id2label: dict):
        self.session = session
        self.tokenizer = tokenizer
        self.id2label = id2label

    def predict(self, texts: list[str]) -> list[str]:
        predictions = []
        for text in texts:
            # Tokenize sequence
            encoded = self.tokenizer.encode(text)
            
            # Prepare ONNX inputs matching the exported graph signature
            inputs = {
                "input_ids": [encoded.ids],
                "attention_mask": [encoded.attention_mask]
            }
            
            # Execute static graph inference
            logits = self.session.run(None, inputs)[0][0]
            
            # Argmax to find the highest scoring class index
            predicted_idx = int(logits.argmax())
            predictions.append(self.id2label[str(predicted_idx)])
            
        return predictions

def fetch_and_load_model() -> ONNXRoutingPipeline:
    """
    Authenticates with the Hugging Face Hub, downloads the serialized ONNX artifact,
    caching it locally, and initializes the inference engine.
    """
    settings.validate_production_settings()

    try:
        logger.info(f"Requesting ONNX graph and tokenizer from {settings.hf_repo_id}")
        
        # Download the required architectural components.
        # hf_hub_download caches these, so subsequent server restarts won't hammer the network.
        onnx_path = hf_hub_download(repo_id=settings.hf_repo_id, filename="model.onnx", token=settings.hf_token)
        tokenizer_path = hf_hub_download(repo_id=settings.hf_repo_id, filename="tokenizer.json", token=settings.hf_token)
        config_path = hf_hub_download(repo_id=settings.hf_repo_id, filename="config.json", token=settings.hf_token)
        
        logger.info("Artifacts retrieved. Initializing lightweight inference engine...")
        
        # Load the label mapping
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            id2label = config.get("id2label", {})

        # Initialize the Rust tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        # Enable truncation to match the training setup (max length 128)
        tokenizer.enable_truncation(max_length=128)
        tokenizer.enable_padding(length=128)

        # Initialize the ONNX runtime session (CPU Execution Provider)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        return ONNXRoutingPipeline(session, tokenizer, id2label)

    except HfHubHTTPError as http_err:
        logger.error(f"Registry Authentication Error: {str(http_err)}")
        raise RuntimeError("Failed to fetch the model from the remote registry. Check HF_TOKEN.") from http_err
        
    except Exception as general_err:
        logger.error(f"An unexpected failure occurred during ONNX engine initialization: {str(general_err)}")
        raise
import logging
import joblib
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from sklearn.pipeline import Pipeline
from api.config import settings

logger = logging.getLogger(__name__)

def fetch_and_load_model() -> Pipeline:
    """
    Authenticates with the Hugging Face Hub, downloads the serialized model artifact,
    caches it locally (saving bandwidth on subsequent loads), and deserializes it into memory.
    """
    settings.validate_production_settings()

    try:
        logger.info(f"Requesting '{settings.model_filename}' from remote registry: {settings.hf_repo_id}")
        
        # hf_hub_download automatically handles local caching. 
        # If the file hasn't changed upstream, it loads from the cache instantly.
        cached_model_path = hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=settings.model_filename,
            token=settings.hf_token
        )
        
        logger.info(f"Artifact retrieved successfully. Deserializing from cache: {cached_model_path}")
        routing_pipeline = joblib.load(cached_model_path)
        
        return routing_pipeline

    except HfHubHTTPError as http_err:
        logger.error(f"Registry Authentication or Not Found Error: {str(http_err)}")
        raise RuntimeError("Failed to fetch the model from the remote registry. Check HF_TOKEN and HF_REPO_ID.") from http_err
        
    except Exception as general_err:
        logger.error(f"An unexpected failure occurred during artifact deserialization: {str(general_err)}")
        raise
import os
import logging
from dataclasses import dataclass
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

@dataclass
class Settings:
    """
    Centralized configuration. 
    Uses lazy evaluation for hf_repo_id to prevent blocking network calls during module import.
    """
    hf_token: str | None = os.getenv("HF_TOKEN")
    
    # Internal cache for the dynamically resolved repo ID
    _hf_repo_id: str | None = os.getenv("HF_REPO_ID")

    @property
    def hf_repo_id(self) -> str:
        # Return the cached or environment-provided ID immediately if it exists
        if self._hf_repo_id:
            return self._hf_repo_id
        
        # Fallback: Resolve dynamically via API only when explicitly accessed
        logger.info("HF_REPO_ID not set in environment. Resolving dynamically via Hugging Face API...")
        api = HfApi(token=self.hf_token)
        username = api.whoami()["name"]
        
        self._hf_repo_id = f"{username}/smart-mail-router"
        return self._hf_repo_id

    def validate_production_settings(self):
        if not self.hf_token:
            logger.warning("HF_TOKEN is missing. The application will fail to pull from private repositories.")

settings = Settings()
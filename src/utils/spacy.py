"""SpaCy model management utilities .

Provides utilities to manage SpaCy models.
Automatically handles model installation and configuration.

Example:
    ```python
    from src.utils.spacy import SpaCyModelManager

    # Check if model is installed
    if not SpaCyModelManager.is_model_installed("en_core_web_sm"):
        SpaCyModelManager.download_model("en_core_web_sm")

    # Or simply set up the model (downloads if needed)
    SpaCyModelManager.setup_spacy_model("en_core_web_sm")
    ```
"""

import subprocess

from loguru import logger
from upath import UPath

from src.utils.config_mngr import global_config


class SpaCyModelManager:
    """Manages SpaCy model installation and configuration."""

    @staticmethod
    def get_model_path(model_name: str) -> UPath:
        """Get the path where the SpaCy model should be stored."""
        path = global_config().get_dir_path(".paths.models", create_if_not_exists=True)
        return path / "spacy_models" / model_name

    @staticmethod
    def is_model_installed(model_name: str) -> bool:
        """Check if the SpaCy model is installed in the specified directory."""
        model_path = SpaCyModelManager.get_model_path(model_name)
        return model_path.exists()

    @staticmethod
    def download_model(model_name: str) -> UPath:
        """Download the SpaCy model if not already present."""
        model_path = SpaCyModelManager.get_model_path(model_name)

        if SpaCyModelManager.is_model_installed(model_name):
            return model_path

        logger.info(f"Downloading SpaCy model '{model_name}'")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)

        return model_path

    @staticmethod
    def setup_spacy_model(model_name: str) -> None:
        """Set up the SpaCy model by downloading it if needed."""
        import subprocess
        
        try:
            import spacy
            # Check if the model can be loaded
            spacy.load(model_name)
            logger.info(f"SpaCy model '{model_name}' is already available")
        except OSError:
            logger.info(f"Downloading SpaCy model '{model_name}'")
            try:
                # Download and install the model
                subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
                # Verify the model is now available
                spacy.load(model_name)
                logger.info(f"SpaCy model '{model_name}' has been successfully installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download SpaCy model '{model_name}': {e}")
                raise
            except OSError as e:
                logger.error(f"Failed to load SpaCy model '{model_name}' after download: {e}")
                raise

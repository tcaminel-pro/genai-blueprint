"""SpaCy model management utilities for Presidio.

Provides utilities to manage SpaCy models for Presidio anonymization.
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

        logger.info(f"Downloading SpaCy model '{model_name}' to {model_path.parent}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name, "--target", str(model_path.parent)], check=True
        )

        return model_path

    @staticmethod
    def setup_spacy_model(model_name: str) -> None:
        """Set up the SpaCy model by downloading it if needed ."""

        if not SpaCyModelManager.is_model_installed(model_name):
            SpaCyModelManager.download_model(model_name)

        # model_path = SpaCyModelManager.get_model_path(model_name)

        # # Set the model path in environment variable for Presidio to use
        # os.environ["PRESIDIO_SPACY_MODEL"] = str(model_path)

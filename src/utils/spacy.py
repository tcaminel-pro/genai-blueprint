"""SpaCy model management utilities for Presidio."""

import os
import subprocess
from pathlib import Path
from typing import Optional


class SpaCyModelManager:
    """Manages SpaCy model installation and configuration for Presidio."""

    DEFAULT_MODEL_NAME = "en_core_web_lg"

    @staticmethod
    def get_model_path(model_name: str | None = None) -> Path:
        """Get the path where the SpaCy model should be stored."""
        model_name = model_name or SpaCyModelManager.DEFAULT_MODEL_NAME
        home_dir = Path.home()
        return home_dir / ".presidio" / "spacy_models" / model_name

    @staticmethod
    def is_model_installed(model_name: str | None = None) -> bool:
        """Check if the SpaCy model is installed in the specified directory."""
        model_name = model_name or SpaCyModelManager.DEFAULT_MODEL_NAME
        model_path = SpaCyModelManager.get_model_path(model_name)
        return model_path.exists()

    @staticmethod
    def download_model(model_name: str | None = None) -> Path:
        """Download the SpaCy model if not already present."""
        model_name = model_name or SpaCyModelManager.DEFAULT_MODEL_NAME
        model_path = SpaCyModelManager.get_model_path(model_name)

        if SpaCyModelManager.is_model_installed(model_name):
            return model_path

        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name, "--target", str(model_path.parent)], check=True
        )

        return model_path

    @staticmethod
    def setup_spacy_model(model_name: str | None = None) -> None:
        """Set up the SpaCy model by downloading it if needed and setting environment variable."""
        model_name = model_name or SpaCyModelManager.DEFAULT_MODEL_NAME

        # Ensure model is available
        if not SpaCyModelManager.is_model_installed(model_name):
            SpaCyModelManager.download_model(model_name)

        model_path = SpaCyModelManager.get_model_path(model_name)

        # Set the model path in environment variable for Presidio to use
        os.environ["PRESIDIO_SPACY_MODEL"] = str(model_path)

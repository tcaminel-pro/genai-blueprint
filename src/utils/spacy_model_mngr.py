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
if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "en_core_web_sm"

    try:
        print(f"Testing SpaCy model setup with: {model_name}")
        SpaCyModelManager.setup_spacy_model(model_name)
        print("✅ Setup successful!")
        print(f"Model installed: {SpaCyModelManager.is_model_installed(model_name)}")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

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

        logger.info(f"Downloading SpaCy model '{model_name}' to {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Use spacy download command without --target first to ensure proper model installation
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name],
            check=True,
        )

        # After downloading, create a symlink in our models directory for consistency
        import spacy

        global_model_path = None
        try:
            # Get the path to the globally installed model
            global_model_path = spacy.util.get_package_path(model_name)
            if global_model_path.exists():
                # Create a symlink to our custom directory
                if not model_path.exists():
                    model_path.symlink_to(global_model_path)
        except Exception as e:
            logger.warning(f"Could not create symlink for model {model_name}: {e}")
            # Fallback to copying the model directory
            if global_model_path and global_model_path.exists():
                import shutil

                shutil.copytree(global_model_path, model_path, dirs_exist_ok=True)

        return model_path

    @staticmethod
    def setup_spacy_model(model_name: str) -> None:
        """Set up the SpaCy model by downloading it if needed."""
        try:
            import spacy

            # Check if model is available globally first
            try:
                spacy.load(model_name)
                logger.info(f"SpaCy model '{model_name}' is available globally")
                return
            except OSError:
                # Model not available globally, need to download
                pass

            # Try to load from our custom directory
            model_path = SpaCyModelManager.get_model_path(model_name)
            if model_path.exists():
                try:
                    spacy.load(model_path)
                    logger.info(f"SpaCy model '{model_name}' loaded from {model_path}")
                    return
                except Exception:
                    # If loading from custom path fails, remove and re-download
                    import shutil

                    if model_path.is_symlink():
                        model_path.unlink()
                    elif model_path.is_dir():
                        shutil.rmtree(model_path)

            # Download the model
            SpaCyModelManager.download_model(model_name)

            # Try loading again
            spacy.load(model_name)
            logger.info(f"SpaCy model '{model_name}' downloaded and loaded successfully")

        except Exception as e:
            logger.error(f"Failed to setup SpaCy model '{model_name}': {e}")
            raise


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "en_core_web_sm"

    print(f"Testing SpaCy model setup with: {model_name}")
    SpaCyModelManager.setup_spacy_model(model_name)
    print("✅ Setup successful!")
    print(f"Model installed: {SpaCyModelManager.is_model_installed(model_name)}")

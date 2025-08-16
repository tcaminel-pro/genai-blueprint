"""Tests for the SpaCy model manager utility."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from upath import UPath

from src.utils.spacy import SpaCyModelManager


class TestSpaCyModelManager:
    """Test cases for SpaCyModelManager class."""

    def test_get_model_path(self, tmp_path: Path) -> None:
        """Test that model path is correctly constructed."""
        with patch("src.utils.spacy.global_config") as mock_config:
            mock_config.return_value.get_dir_path.return_value = UPath(tmp_path)
            
            model_name = "en_core_web_sm"
            expected_path = UPath(tmp_path) / "spacy_models" / model_name
            
            result = SpaCyModelManager.get_model_path(model_name)
            
            assert result == expected_path
            mock_config.return_value.get_dir_path.assert_called_once_with(".paths.models", create_if_not_exists=True)

    def test_is_model_installed_true(self, tmp_path: Path) -> None:
        """Test is_model_installed returns True when model exists."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        with patch("src.utils.spacy.SpaCyModelManager.get_model_path", return_value=UPath(model_path)):
            assert SpaCyModelManager.is_model_installed(model_name) is True

    def test_is_model_installed_false(self, tmp_path: Path) -> None:
        """Test is_model_installed returns False when model doesn't exist."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        
        with patch("src.utils.spacy.SpaCyModelManager.get_model_path", return_value=UPath(model_path)):
            assert SpaCyModelManager.is_model_installed(model_name) is False

    @patch("subprocess.run")
    def test_download_model_not_installed(self, mock_subprocess: MagicMock, tmp_path: Path) -> None:
        """Test download_model when model is not installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        
        with patch("src.utils.spacy.SpaCyModelManager.get_model_path", return_value=UPath(model_path)):
            with patch("src.utils.spacy.SpaCyModelManager.is_model_installed", return_value=False):
                result = SpaCyModelManager.download_model(model_name)
                
                assert result == UPath(model_path)
                mock_subprocess.assert_called_once_with(
                    ["python", "-m", "spacy", "download", model_name, "--target", str(tmp_path / "spacy_models")],
                    check=True
                )

    @patch("subprocess.run")
    def test_download_model_already_installed(self, mock_subprocess: MagicMock, tmp_path: Path) -> None:
        """Test download_model when model is already installed."""
        model_name = "en_core_web_sm"
        model_path = tmp_path / "spacy_models" / model_name
        
        with patch("src.utils.spacy.SpaCyModelManager.get_model_path", return_value=UPath(model_path)):
            with patch("src.utils.spacy.SpaCyModelManager.is_model_installed", return_value=True):
                result = SpaCyModelManager.download_model(model_name)
                
                assert result == UPath(model_path)
                mock_subprocess.assert_not_called()

    @patch("src.utils.spacy.SpaCyModelManager.download_model")
    @patch("src.utils.spacy.SpaCyModelManager.is_model_installed")
    def test_setup_spacy_model_not_installed(self, mock_is_installed: MagicMock, mock_download: MagicMock) -> None:
        """Test setup_spacy_model when model is not installed."""
        model_name = "en_core_web_sm"
        
        mock_is_installed.return_value = False
        
        SpaCyModelManager.setup_spacy_model(model_name)
        
        mock_is_installed.assert_called_once_with(model_name)
        mock_download.assert_called_once_with(model_name)

    @patch("src.utils.spacy.SpaCyModelManager.download_model")
    @patch("src.utils.spacy.SpaCyModelManager.is_model_installed")
    def test_setup_spacy_model_already_installed(self, mock_is_installed: MagicMock, mock_download: MagicMock) -> None:
        """Test setup_spacy_model when model is already installed."""
        model_name = "en_core_web_sm"
        
        mock_is_installed.return_value = True
        
        SpaCyModelManager.setup_spacy_model(model_name)
        
        mock_is_installed.assert_called_once_with(model_name)
        mock_download.assert_not_called()

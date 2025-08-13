"""Tests for the Config class and its configuration management features.

These tests verify:
- Basic configuration loading and retrieval
- Environment variable substitution
- Runtime modifications
- Configuration section switching
- Error handling for missing keys
"""

import os
import tempfile
from pathlib import Path
from unittest import TestCase

import pytest

from src.utils.config_mngr import OmegaConfig, global_config, global_config_reload


def test_singleton() -> None:
    """Verify the singleton pattern works correctly."""
    config1 = OmegaConfig.singleton()
    config2 = OmegaConfig.singleton()
    assert config1 is config2


def test_config_section_switch() -> None:
    """Verify switching between configuration sections works."""
    config = OmegaConfig.singleton()
    original_value = config.get_str("llm.default_model")

    # Switch to a different config section
    config.select_config("training_azure")
    new_value = config.get_str("llm.default_model")

    assert new_value != original_value


def test_runtime_modification() -> None:
    """Verify runtime modifications to config values."""
    config = OmegaConfig.singleton()
    config.set("test.temp_value", "initial")
    assert config.get_str("test.temp_value") == "initial"

    # Modify the value
    config.set("test.temp_value", "modified")
    assert config.get_str("test.temp_value") == "modified"


def test_missing_key() -> None:
    """Verify error handling for missing configuration keys."""
    config = OmegaConfig.singleton()
    with pytest.raises(ValueError):
        config.get_str("nonexistent.key")


def test_get_list() -> None:
    """Verify retrieval of list values from config."""
    config = OmegaConfig.singleton()

    result = config.get_list("chains.modules")
    assert isinstance(result, list)


class TestOmegaConfig(TestCase):
    """Test cases for OmegaConfig configuration management."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary directory for test config files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.yaml"

        # Create test configuration content
        self.test_config_content = """                                                                     
default_config: test_env                                                                                   
paths:                                                                                                     
  data_root: /tmp/test_data                                                                                
  project: /tmp/test_project                                                                               
  models: /tmp/models                                                                                      
test_env:                                                                                                  
  llm:                                                                                                     
    default_model: "gpt-3.5-turbo"                                                                         
    max_tokens: 1000                                                                                       
    temperature: 0.7                                                                                       
  features:                                                                                                
    enable_caching: true                                                                                   
    enable_logging: false                                                                                  
  db:                                                                                                      
    host: "localhost"                                                                                      
    port: 5432                                                                                             
    name: "test_db"                                                                                        
prod_env:                                                                                                  
  llm:                                                                                                     
    default_model: "gpt-4"                                                                                 
    max_tokens: 2000                                                                                       
    temperature: 0.1                                                                                       
  features:                                                                                                
    enable_caching: true                                                                                   
    enable_logging: true                                                                                   
  db:                                                                                                      
    host: "prod.example.com"                                                                               
    port: 5432                                                                                             
    name: "prod_db"                                                                                   
cli:                                                                                                       
  commands:                                                                                                
    - test.module:register_commands                                                                        
    - test.module2:register_commands                                                                      
ui:                                                                                                        
  app_name: "Test App"                                                                                     
  pages_dir: /tmp/pages                                                                                    
"""

        # Write test config to temporary file
        self.config_path.write_text(self.test_config_content)

        # Set environment variable for testing
        os.environ["BLUEPRINT_CONFIG"] = "test_env"

        # Create test instance
        self.config = OmegaConfig.create(self.config_path)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        # Clean up environment
        if "BLUEPRINT_CONFIG" in os.environ:
            del os.environ["BLUEPRINT_CONFIG"]

    def test_create_config(self) -> None:
        """Test configuration loading and creation."""
        self.assertIsInstance(self.config, OmegaConfig)
        self.assertEqual(self.config.selected_config, "test_env")
        self.assertIn("test_env", self.config.root)

    def test_get_string_value(self) -> None:
        """Test getting string configuration values."""
        model = self.config.get("llm.default_model")
        self.assertEqual(model, "gpt-3.5-turbo")

        # Test with default value
        missing_value = self.config.get("nonexistent.key", "default_value")
        self.assertEqual(missing_value, "default_value")

    def test_get_int_value(self) -> None:
        """Test getting integer configuration values."""
        max_tokens = self.config.get("llm.max_tokens")
        self.assertEqual(max_tokens, 1000)

        port = self.config.get("db.port")
        self.assertEqual(port, 5432)

    def test_get_float_value(self) -> None:
        """Test getting float configuration values."""
        temperature = self.config.get("llm.temperature")
        self.assertEqual(temperature, 0.7)

    def test_get_bool_value(self) -> None:
        """Test getting boolean configuration values."""
        enable_caching = self.config.get("features.enable_caching")
        self.assertTrue(enable_caching)

        enable_logging = self.config.get("features.enable_logging")
        self.assertFalse(enable_logging)

    def test_get_str_method(self) -> None:
        """Test get_str type-safe method."""
        model = self.config.get_str("llm.default_model")
        self.assertEqual(model, "gpt-3.5-turbo")

        with self.assertRaises(TypeError):
            self.config.get_str("llm.max_tokens")

    def test_get_bool_method(self) -> None:
        """Test get_bool type-safe method."""
        enable_caching = self.config.get_bool("features.enable_caching")
        self.assertTrue(enable_caching)

        # Test string boolean conversion
        os.environ["TEST_BOOL"] = "true"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""                                                                                    
test_bool_env:                                                                                             
  str_true: "${oc.env:TEST_BOOL}"                                                                          
""")
            f.flush()
            test_config = OmegaConfig.create(Path(f.name))
            os.unlink(f.name)

        bool_value = test_config.get_bool("str_true")
        self.assertTrue(bool_value)

    def test_get_list_method(self) -> None:
        """Test get_list type-safe method."""
        commands = self.config.get_list("cli.commands")
        self.assertEqual(commands, ["test.module:register_commands", "test.module2:register_commands"])

        with self.assertRaises(TypeError):
            self.config.get_list("llm.default_model")

    def test_get_dict_method(self) -> None:
        """Test get_dict type-safe method."""
        db_config = self.config.get_dict("db")
        expected = {"host": "localhost", "port": 5432, "name": "test_db"}
        self.assertEqual(db_config, expected)

        # Test with expected keys validation
        db_config_validated = self.config.get_dict("db", expected_keys=["host", "port", "name"])
        self.assertEqual(db_config_validated, expected)

        with self.assertRaises(KeyError):
            self.config.get_dict("db", expected_keys=["host", "missing_key"])

    def test_select_config(self) -> None:
        """Test switching between configuration environments."""
        # Initially in test_env
        self.assertEqual(self.config.get("llm.default_model"), "gpt-3.5-turbo")
        self.assertEqual(self.config.get("llm.max_tokens"), 1000)

        # Switch to prod_env
        self.config.select_config("prod_env")
        self.assertEqual(self.config.get("llm.default_model"), "gpt-4")
        self.assertEqual(self.config.get("llm.max_tokens"), 2000)

        # Test switching to non-existent config
        with self.assertRaises(ValueError):
            self.config.select_config("nonexistent_env")

    def test_set_runtime_override(self) -> None:
        """Test setting runtime configuration overrides."""
        original_model = self.config.get("llm.default_model")
        self.assertEqual(original_model, "gpt-3.5-turbo")

        # Set override
        self.config.set("llm.default_model", "custom-model")
        self.assertEqual(self.config.get("llm.default_model"), "custom-model")

        # Test setting nested override
        self.config.set("new.nested.value", "test")
        self.assertEqual(self.config.get("new.nested.value"), "test")

    def test_get_dir_path(self) -> None:
        """Test getting directory paths."""
        # Test with existing directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config.set("test_dir", temp_dir)
            dir_path = self.config.get_dir_path("test_dir")
            self.assertEqual(str(dir_path), temp_dir)

        # Test creating non-existent directory
        temp_path = Path(self.temp_dir.name) / "new_dir"
        self.config.set("new_dir", str(temp_path))
        dir_path = self.config.get_dir_path("new_dir", create_if_not_exists=True)
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())

    def test_get_file_path(self) -> None:
        """Test getting file paths."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("test content")
            temp_path = temp_file.name

        try:
            self.config.set("test_file", temp_path)
            file_path = self.config.get_file_path("test_file")
            self.assertEqual(str(file_path), temp_path)

            # Test non-existent file
            self.config.set("missing_file", "/nonexistent/path/file.txt")
            with self.assertRaises(FileNotFoundError):
                self.config.get_file_path("missing_file")
        finally:
            os.unlink(temp_path)

    def test_merge_with_additional_config(self) -> None:
        """Test merging additional configuration files."""
        additional_config = """                                                                            
additional_env:                                                                                            
  new_setting: "from_additional"                                                                           
  llm:                                                                                                     
    provider: "openai"                                                                                     
"""
        additional_path = Path(self.temp_dir.name) / "additional.yaml"
        additional_path.write_text(additional_config)

        # Merge additional config
        self.config.merge_with(str(additional_path))
        self.assertIn("additional_env", self.config.root)
        self.assertEqual(self.config.get("additional_env.new_setting"), "from_additional")

    def test_dsn_generation(self) -> None:
        """Test DSN URL generation for database connections."""
        test_dsn = "postgresql://user:pass@localhost:5432/testdb"
        self.config.set("test_dsn", test_dsn)

        dsn = self.config.get_dsn("test_dsn")
        self.assertEqual(dsn, test_dsn)

        # Test with driver override
        dsn_with_driver = self.config.get_dsn("test_dsn", driver="asyncpg")
        self.assertEqual(dsn_with_driver, "postgresql+asyncpg://user:pass@localhost:5432/testdb")

    def test_environment_variable_interpolation(self) -> None:
        """Test OmegaConf environment variable interpolation."""
        os.environ["TEST_VAR"] = "interpolated_value"

        interp_config = """                                                                                
test_interp:                                                                                               
  value: "${oc.env:TEST_VAR}"                                                                              
"""
        interp_path = Path(self.temp_dir.name) / "interp.yaml"
        interp_path.write_text(interp_config)

        config = OmegaConfig.create(interp_path)
        value = config.get("test_interp.value")
        self.assertEqual(value, "interpolated_value")

    def test_global_config_singleton(self) -> None:
        """Test the global config singleton."""
        config1 = global_config()
        config2 = global_config()

        # Should be the same instance
        self.assertIs(config1, config2)

        # Test reload
        global_config_reload()
        config3 = global_config()
        self.assertIsNot(config1, config3)

    def test_invalid_config_file(self) -> None:
        """Test handling of invalid configuration files."""
        invalid_path = Path(self.temp_dir.name) / "invalid.yaml"
        invalid_path.write_text("invalid: yaml: content: [")

        with self.assertRaises(Exception):
            OmegaConfig.create(invalid_path)

    def test_missing_config_section(self) -> None:
        """Test handling of missing configuration sections."""
        with self.assertRaises(ValueError):
            self.config.get("missing_section.key")

    def test_nested_access(self) -> None:
        """Test deeply nested configuration access."""
        nested_config = """                                                                                
deep:                                                                                                      
  nested:                                                                                                  
    structure:                                                                                             
      value: 42                                                                                            
      list: [1, 2, 3]                                                                                      
      dict:                                                                                                
        inner: "test"                                                                                      
"""
        nested_path = Path(self.temp_dir.name) / "nested.yaml"
        nested_path.write_text(nested_config)

        config = OmegaConfig.create(nested_path)

        # Test deep access
        value = config.get("deep.nested.structure.value")
        self.assertEqual(value, 42)

        nested_list = config.get_list("deep.nested.structure.list")
        self.assertEqual(nested_list, [1, 2, 3])

        nested_dict = config.get_dict("deep.nested.structure.dict")
        self.assertEqual(nested_dict, {"inner": "test"})


if __name__ == "__main__":
    import unittest

    unittest.main()

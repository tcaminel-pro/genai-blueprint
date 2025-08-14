"""Unit tests for kv_store module."""

import os
import unittest
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from src.utils.pydantic.kv_store import (
    _encode_to_alphanumeric,
    load_object_from_kvstore,
    save_object_to_kvstore,
)


class SampleModel(BaseModel):
    """Test model for kv_store testing."""

    name: str
    value: int


class TestKVStore(unittest.TestCase):
    """Test cases for key-value store functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_model = SampleModel(name="test_object", value=42)

    def test_encode_to_alphanumeric(self):
        """Test alphanumeric encoding function."""
        # Test basic encoding
        self.assertEqual(_encode_to_alphanumeric("hello world"), "hello_world")
        self.assertEqual(_encode_to_alphanumeric("test-file.json"), "test-file.json")

        # Test special characters
        self.assertEqual(_encode_to_alphanumeric("test@#$%^&*()"), "test_________")

        # Test unicode
        self.assertEqual(_encode_to_alphanumeric("café"), "cafe")

    def test_file_storage_basic(self):
        """Test basic file-based storage functionality."""
        with TemporaryDirectory() as temp_dir:
            # Configure file storage to use temp directory
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            # Test saving and loading
            key = "test_key"
            save_object_to_kvstore(key, self.test_model, "file")

            retrieved = load_object_from_kvstore(SampleModel, key, "file")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.name, "test_object")
            self.assertEqual(retrieved.value, 42)

    def test_file_storage_nonexistent_key(self):
        """Test loading non-existent key returns None."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            retrieved = load_object_from_kvstore(SampleModel, "nonexistent", "file")
            self.assertIsNone(retrieved)

    def test_file_storage_dict_key(self):
        """Test storage with dictionary keys."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            dict_key = {"user_id": 123, "session": "abc"}
            save_object_to_kvstore(dict_key, self.test_model, "file")

            retrieved = load_object_from_kvstore(SampleModel, dict_key, "file")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.name, "test_object")

    def test_file_storage_overwrite(self):
        """Test overwriting existing key."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            key = "test_key"
            save_object_to_kvstore(key, self.test_model, "file")

            # Create new model and overwrite
            new_model = SampleModel(name="updated", value=99)
            save_object_to_kvstore(key, new_model, "file")

            retrieved = load_object_from_kvstore(SampleModel, key, "file")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.name, "updated")
            self.assertEqual(retrieved.value, 99)

    def test_file_storage_multiple_models(self):
        """Test storing different model types."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            class AnotherModel(BaseModel):
                title: str
                count: float

            model1 = SampleModel(name="model1", value=1)
            model2 = AnotherModel(title="model2", count=2.5)

            save_object_to_kvstore("key1", model1, "file")
            save_object_to_kvstore("key2", model2, "file")

            retrieved1 = load_object_from_kvstore(SampleModel, "key1", "file")
            retrieved2 = load_object_from_kvstore(AnotherModel, "key2", "file")

            self.assertIsNotNone(retrieved1)
            self.assertIsNotNone(retrieved2)
            self.assertEqual(retrieved1.name, "model1")
            self.assertEqual(retrieved2.title, "model2")

    def test_sql_storage_sqlite(self):
        """Test SQL storage with SQLite (fallback for PostgreSQL)."""

        with TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_kv_store.db")
            db_url = f"sqlite:///{db_path}"

            # Configure SQL storage to use SQLite
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.sql.path", db_url)

            try:
                model = SampleModel(name="sql_test", value=123)
                save_object_to_kvstore("sql_key", model, "sql")

                retrieved = load_object_from_kvstore(SampleModel, "sql_key", "sql")
                self.assertIsNotNone(retrieved)
                self.assertEqual(retrieved.name, "sql_test")
                self.assertEqual(retrieved.value, 123)

            except ImportError as e:
                # Skip if SQL dependencies not available
                self.skipTest(f"SQL storage dependencies not available: {e}")

    def test_invalid_key_type(self):
        """Test that invalid key types raise appropriate errors."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            with self.assertRaises(ValueError):
                save_object_to_kvstore(123, self.test_model, "file")  # type: ignore

    def test_invalid_store_id(self):
        """Test that invalid store_id raises appropriate errors."""
        with TemporaryDirectory() as temp_dir:
            from src.utils.config_mngr import global_config

            global_config().set("kv_store.file.path", temp_dir)

            with self.assertRaises(ValueError):
                save_object_to_kvstore("key", self.test_model, "invalid_store")


if __name__ == "__main__":
    unittest.main()

"""Key-value storage factory for Pydantic objects.

Provides a factory pattern for creating and managing key-value stores for
Pydantic objects with support for multiple storage backends including
file-based and SQL storage (PostgreSQL).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from langchain.storage import LocalFileStore
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator
from unidecode import unidecode

try:
    from langchain_community.storage import SQLStore

    HAS_SQL_STORE = True
except ImportError:
    HAS_SQL_STORE = False
    SQLStore = None

from src.utils.config_mngr import global_config
from src.utils.singleton import once

T = TypeVar("T", bound=BaseModel)

StorageBackend = Literal["file", "sql"]
StorageEngine = Literal["LocalFileStore", "SQLStore"]


def encode_to_alphanumeric(input_string: str) -> str:
    """Encode a string to alphanumeric by first transliterating it to ASCII, then replace non alphanumeric char (plus . and -) by _."""
    ascii_string = unidecode(input_string)
    encoded_string = re.sub(r"[^a-zA-Z0-9_.-]", "_", ascii_string)
    return encoded_string


def _encode_key(key: str | dict) -> str:
    """Encode key for storage compatibility."""
    if isinstance(key, str):
        encoded_key = encode_to_alphanumeric(key)
    elif isinstance(key, dict):
        hash_object = hashlib.md5()
        hash_object.update(json.dumps(key, sort_keys=True).encode())
        encoded_key = hash_object.hexdigest()
    else:
        raise ValueError("key need to be str or dict")
    return encoded_key


class StorageInfo(BaseModel):
    """Information about a storage backend configuration.

    Attributes:
        id: Unique identifier for the storage backend
        backend: Storage backend type ("file" or "sql")
        engine: Storage engine implementation
        config: Configuration dictionary for the storage backend
    """

    id: str
    backend: StorageBackend
    engine: StorageEngine
    config: dict = {}


class KVStoreFactory(BaseModel):
    """Factory for creating and managing key-value stores for Pydantic objects.

    Handles creation of storage backends with appropriate configuration
    based on storage type and provider.

    Attributes:
        store_id: Unique identifier for the storage backend
    """

    store_id: str = Field(default=None, validate_default=True)

    @field_validator("store_id", mode="before")
    def check_known(cls, store_id: str | None) -> str:
        """Validate storage backend identifier."""
        if store_id is None:
            store_id = global_config().get_str("kv_store.default_backend", default="file")
        if store_id not in KVStoreFactory.known_items():
            raise ValueError(f"Unknown storage backend: {store_id}")
        return store_id

    @computed_field
    @property
    def info(self) -> StorageInfo:
        """Return storage backend configuration."""
        return KVStoreFactory.known_items_dict().get(self.store_id)  # type: ignore

    @staticmethod
    def _read_storage_list_file() -> list[StorageInfo]:
        """Read storage configuration from YAML file."""
        yml_file = global_config().get_file_path("kv_store.list")
        try:
            with open(yml_file) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback to default configuration
            data = {
                "backends": [
                    {"id": "file", "backend": "file", "engine": "LocalFileStore", "config": {"path": "kv_store"}},
                    {
                        "id": "sql",
                        "backend": "sql",
                        "engine": "SQLStore",
                        "config": {"url": "postgres.url", "namespace_prefix": "kv_store"},
                    },
                ]
            }

        backends = []
        for backend_entry in data.get("backends", []):
            backends.append(StorageInfo(**backend_entry))
        return backends

    @once
    @staticmethod
    def known_list() -> list[StorageInfo]:
        """List all known storage backends."""
        return KVStoreFactory._read_storage_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, StorageInfo]:
        """Create a dictionary of available storage backends."""
        return {item.id: item for item in KVStoreFactory.known_list()}

    @staticmethod
    def known_items() -> list[str]:
        """List identifiers of available storage backends."""
        return sorted(KVStoreFactory.known_items_dict().keys())

    def get_store(self, model_class: type[T]) -> "BaseKVStore":
        """Create a key-value store instance for the given model class."""
        if self.info.backend == "file":
            return FileKVStore(model_class, self.info)
        elif self.info.backend == "sql":
            return SQLKVStore(model_class, self.info)
        else:
            raise ValueError(f"Unsupported storage backend: {self.info.backend}")


class BaseKVStore:
    """Base class for key-value storage implementations."""

    def __init__(self, model_class: type[T], storage_info: StorageInfo):
        self.model_class = model_class
        self.storage_info = storage_info
        self.namespace = model_class.__name__

    def save(self, key: str | dict, obj: BaseModel) -> None:
        """Save a Pydantic object to the store."""
        raise NotImplementedError

    def load(self, key: str | dict) -> T | None:
        """Load a Pydantic object from the store."""
        raise NotImplementedError


class FileKVStore(BaseKVStore):
    """File-based key-value store using LocalFileStore."""

    def __init__(self, model_class: type[T], storage_info: StorageInfo):
        super().__init__(model_class, storage_info)
        path_key = storage_info.config.get("path", "kv_store.path")
        self.file_store_path = global_config().get_dir_path(path_key, create_if_not_exists=True)
        self.dir_path = self.file_store_path / self.namespace
        self.store = LocalFileStore(self.dir_path)

    def save(self, key: str | dict, obj: BaseModel) -> None:
        """Save a Pydantic object to file storage."""
        encoded_key = _encode_key(key) + ".json"
        obj_bytes = obj.model_dump_json().encode("utf-8")
        self.store.mset([(encoded_key, obj_bytes)])
        logger.debug(f"Saved {self.namespace}/{encoded_key} to file storage")

    def load(self, key: str | dict) -> T | None:
        """Load a Pydantic object from file storage."""
        encoded_key = _encode_key(key) + ".json"
        stored_bytes = self.store.mget([encoded_key])[0]

        if not stored_bytes:
            return None

        try:
            logger.debug(f"Loaded {self.namespace}/{encoded_key} from file storage")
            return self.model_class.model_validate_json(stored_bytes.decode("utf-8"))
        except ValidationError as ex:
            logger.warning(f"Failed to load JSON for {self.namespace}/{encoded_key}: {ex}")
            return None
        except Exception as ex:
            logger.warning(f"Failed to load {self.namespace}/{encoded_key}: {ex}")
            return None


class SQLKVStore(BaseKVStore):
    """SQL-based key-value store using SQLStore."""

    def __init__(self, model_class: type[T], storage_info: StorageInfo):
        super().__init__(model_class, storage_info)

        if not HAS_SQL_STORE:
            raise ImportError("langchain_community is required for SQL storage")

        url_key = storage_info.config.get("url", "postgres.url")
        namespace_prefix = storage_info.config.get("namespace_prefix", "kv_store")

        db_url = global_config().get_str(url_key)
        if not db_url:
            raise ValueError(f"Database URL not found in configuration: {url_key}")

        self.namespace = f"{namespace_prefix}_{self.namespace}"
        self.store = SQLStore(namespace=self.namespace, db_url=db_url)
        self.store.create_schema()

    def save(self, key: str | dict, obj: BaseModel) -> None:
        """Save a Pydantic object to SQL storage."""
        encoded_key = _encode_key(key)
        obj_bytes = obj.model_dump_json().encode("utf-8")
        self.store.mset([(encoded_key, obj_bytes)])
        logger.debug(f"Saved {self.namespace}/{encoded_key} to SQL storage")

    def load(self, key: str | dict) -> T | None:
        """Load a Pydantic object from SQL storage."""
        encoded_key = _encode_key(key)
        stored_bytes = self.store.mget([encoded_key])[0]

        if not stored_bytes:
            return None

        try:
            logger.debug(f"Loaded {self.namespace}/{encoded_key} from SQL storage")
            return self.model_class.model_validate_json(stored_bytes.decode("utf-8"))
        except ValidationError as ex:
            logger.warning(f"Failed to load JSON for {self.namespace}/{encoded_key}: {ex}")
            return None
        except Exception as ex:
            logger.warning(f"Failed to load {self.namespace}/{encoded_key}: {ex}")
            return None


# Convenience functions for backward compatibility
def save_object_to_kvstore(key: str | dict, obj: BaseModel, store_id: str | None = None) -> None:
    """Save a Pydantic object to the default key-value store."""
    store = KVStoreFactory(store_id=store_id).get_store(type(obj))
    store.save(key, obj)


def load_object_from_kvstore(model_class: type[T], key: str | dict, store_id: str | None = None) -> T | None:
    """Load a Pydantic object from the default key-value store."""
    store = KVStoreFactory(store_id=store_id).get_store(model_class)
    return store.load(key)


# Quick test
if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    class TestModel(BaseModel):
        name: str
        value: int

    # Test the factory pattern
    print("Testing KVStoreFactory...")

    # Test file storage
    print("\n1. Testing file storage:")
    file_store = KVStoreFactory(store_id="file").get_store(TestModel)
    test_model = TestModel(name="test_object", value=42)
    file_store.save("test_key", test_model)
    retrieved = file_store.load("test_key")
    print(f"   File storage result: {retrieved}")

    # Test SQL storage (if PostgreSQL URL is configured)
    print("\n2. Testing SQL storage:")
    try:
        sql_store = KVStoreFactory(store_id="sql").get_store(TestModel)
        sql_model = TestModel(name="sql_test_object", value=123)
        sql_store.save("sql_test_key", sql_model)
        sql_retrieved = sql_store.load("sql_test_key")
        print(f"   SQL storage result: {sql_retrieved}")
    except Exception as e:
        print(f"   SQL storage failed: {e}")

    # Test backward compatibility functions
    print("\n3. Testing backward compatibility:")
    save_object_to_kvstore("compat_key", test_model)
    compat_retrieved = load_object_from_kvstore(TestModel, "compat_key")
    print(f"   Backward compatibility result: {compat_retrieved}")

    # Test with dict keys
    print("\n4. Testing dict keys:")
    dict_key = {"user_id": 123, "session": "abc"}
    save_object_to_kvstore(dict_key, test_model)
    dict_retrieved = load_object_from_kvstore(TestModel, dict_key)
    print(f"   Dict key result: {dict_retrieved}")

    print("\nAll tests completed!")

"""Local file-based and SQL storage for Pydantic objects.

Provides functions to save and retrieve Pydantic object using a key-value store.
Supports both file-based storage and SQL storage (PostgreSQL via SQLStore).
Keys are automatically encoded for filesystem compatibility.
Supports both string and dictionary keys.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, TypeVar

from langchain.storage import LocalFileStore
try:
    from langchain_community.storage import SQLStore
    HAS_SQL_STORE = True
except ImportError:
    HAS_SQL_STORE = False
    SQLStore = None
from loguru import logger
from pydantic import BaseModel, ValidationError
from unidecode import unidecode

from src.utils.config_mngr import global_config

T = TypeVar("T", bound=BaseModel)

StorageBackend = Literal["file", "sql"]

# TODO : Consider using EncoderBackedStore


def encode_to_alphanumeric(input_string: str) -> str:
    """Encode a string to alphanumeric by first transliterating  it to ASCII, then replace  non alphanumeric char (plus . and -) by _."""
    ascii_string = unidecode(input_string)
    encoded_string = re.sub(r"[^a-zA-Z0-9_.-]", "_", ascii_string)
    return encoded_string


def _encode_key(key: str | dict) -> str:
    if isinstance(key, str):
        encoded_key = encode_to_alphanumeric(key)
    elif isinstance(key, dict):
        hash_object = hashlib.md5()
        hash_object.update(json.dumps(key, sort_keys=True).encode())
        encoded_key = hash_object.hexdigest()
    else:
        raise ValueError("key need to be str or dict")
    return encoded_key + ".json"  # add json so the file can be easily viewed


def save_object_to_kvstore(
    key: str | dict, 
    obj: BaseModel, 
    file_store_path: Path | None = None,
    backend: StorageBackend = "file"
) -> None:
    """Save a Pydantic model to a key-value store.

    The model can be saved to either a local file-based store or a SQL database.
    For file storage, the model is saved to a directory based on the model's class name.
    For SQL storage, PostgreSQL is used with the connection string from configuration.

    Args:
        key: Unique identifier for the object (str or dict). If dict, its hash is used.
        obj: Pydantic model instance to save.
        file_store_path: Root directory for file storage (used only for file backend).
        backend: Storage backend to use ("file" or "sql").
    """
    class_name = obj.__class__.__name__
    obj_bytes = obj.model_dump_json().encode("utf-8")
    encoded_key = _encode_key(key)

    if backend == "sql":
        if not HAS_SQL_STORE:
            raise ImportError("langchain_community is required for SQL storage")
        
        namespace = f"kv_store_{class_name}"
        postgres_url = global_config().get_str("postgres.url")
        if not postgres_url:
            raise ValueError("PostgreSQL URL not found in configuration")
        
        sql_store = SQLStore(namespace=namespace, db_url=postgres_url)
        sql_store.create_schema()
        sql_store.mset([(encoded_key, obj_bytes)])
        logger.debug(f"add key '{namespace}/{encoded_key}' to SQL kv_store")
    else:  # file backend
        if file_store_path is None:
            file_store_path = global_config().get_dir_path("kv_store.path")
        
        dir_root_name = file_store_path / class_name
        file_store = LocalFileStore(dir_root_name)
        file_store.mset([(encoded_key, obj_bytes)])
        logger.debug(f"add key '{class_name}/{encoded_key}' to local kv_store {file_store_path}'")


def load_object_from_kvstore(
    model_class: type[T], 
    key: str | dict, 
    file_store_path: Path | None = None,
    backend: StorageBackend = "file"
) -> T | None:
    """Read a Pydantic object from a key-value store.

    Args:
        model_class: Pydantic model class to reconstruct.
        key: Unique identifier for the stored object (str or dict). If dict, its hash is used.
        file_store_path: Root directory for file storage (used only for file backend).
        backend: Storage backend to use ("file" or "sql").

    Returns:
        Instance of the specified Pydantic model, or None if not found.
    """
    class_name = model_class.__name__
    encoded_key = _encode_key(key)

    if backend == "sql":
        if not HAS_SQL_STORE:
            raise ImportError("langchain_community is required for SQL storage")
        
        namespace = f"kv_store_{class_name}"
        postgres_url = global_config().get_str("postgres.url")
        if not postgres_url:
            raise ValueError("PostgreSQL URL not found in configuration")
        
        sql_store = SQLStore(namespace=namespace, db_url=postgres_url)
        stored_bytes = sql_store.mget([encoded_key])[0]
    else:  # file backend
        if file_store_path is None:
            file_store_path = global_config().get_dir_path("kv_store.path", create_if_not_exists=True)
        
        file_store = LocalFileStore(file_store_path / class_name)
        stored_bytes = file_store.mget([encoded_key])[0]

    if not stored_bytes:
        return None
    
    try:
        logger.debug(f"read '{class_name}/{encoded_key}' from {backend} KV store")
        return model_class.model_validate_json(stored_bytes.decode("utf-8"))
    except ValidationError as ex:
        logger.warning(f"failed to load JSON value for {class_name}/{encoded_key}. Error is : {ex}")
        return None
    except Exception as ex:
        logger.warning(f"failed to load JSON value for {class_name}/{encoded_key}. Exception is : {ex}")
        return None


# Quick test
if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    class TestModel(BaseModel):
        name: str
        value: int

    # Test file-based storage
    print("Testing file-based storage...")
    with TemporaryDirectory(delete=False) as temp_dir:
        temp_path = Path(temp_dir)
        test_model = TestModel(name="test_object", value=42)
        
        save_object_to_kvstore("unique_key", test_model, temp_path, backend="file")
        retrieved_model = load_object_from_kvstore(TestModel, "unique_key", temp_path, backend="file")
        print(f"File storage test: {retrieved_model}")

    # Test SQL storage (if PostgreSQL URL is configured)
    try:
        print("\nTesting SQL storage...")
        postgres_url = global_config().get_str("postgres.url")
        if postgres_url:
            test_model_sql = TestModel(name="sql_test_object", value=123)
            save_object_to_kvstore("sql_unique_key", test_model_sql, backend="sql")
            retrieved_sql_model = load_object_from_kvstore(TestModel, "sql_unique_key", backend="sql")
            print(f"SQL storage test: {retrieved_sql_model}")
        else:
            print("PostgreSQL URL not configured - skipping SQL storage test")
    except Exception as e:
        print(f"SQL storage test failed: {e}")

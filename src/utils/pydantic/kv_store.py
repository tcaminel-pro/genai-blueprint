"""Key_value storage for Pydantic objects.

Provides functions to save and retrieve Pydantic object using a key-value store.
Keys are automatically encoded for filesystem compatibility.
Supports both string and dictionary keys.
"""

import hashlib
import json
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError
from unidecode import unidecode

from src.ai_extra.kv_store_factory import KvStoreFactory

T = TypeVar("T", bound=BaseModel)

# TODO : Consider using EncoderBackedStore


class StoredObject(BaseModel):
    """Wrapper for storing Pydantic objects with metadata."""

    content: dict  # Store as dict to allow proper deserialization
    metadata: dict = {}


def _encode_to_alphanumeric(input_string: str) -> str:
    """Encode a string to alphanumeric by first transliterating  it to ASCII, then replace  non alphanumeric char (plus . and -) by _."""
    ascii_string = unidecode(input_string)
    encoded_string = re.sub(r"[^a-zA-Z0-9_.-]", "_", ascii_string)
    return encoded_string


def _encode_key(key: str | dict) -> str:
    if isinstance(key, str):
        encoded_key = _encode_to_alphanumeric(key)
    elif isinstance(key, dict):
        hash_object = hashlib.md5()
        hash_object.update(json.dumps(key, sort_keys=True).encode())
        encoded_key = hash_object.hexdigest()
    else:
        raise ValueError("key need to be str or dict")
    return encoded_key + ".json"  # add json so the file can be easily viewed


def save_object_to_kvstore(
    key: str | dict, obj: BaseModel, kv_store_id: str = "file", metadata: dict | None = None
) -> None:
    """Save a Pydantic model to a local file-based key-value store.

    The model is saved to a directory based on the model's class name. The key is
    encoded to ensure filesystem compatibility.

    Args:
        key: Unique identifier for the object (str or dict). If dict, its hash is used.
        obj: Pydantic model instance to save.
        kv_store_id: Identifier for the storage backend.
        metadata: Optional dictionary of metadata to store with the object.
    """
    # Use the lowercase class name of the Pydantic model if no config_key is provided
    class_name = obj.__class__.__name__
    kv_store = KvStoreFactory(id=kv_store_id, root=class_name).get()

    # Create a wrapper that contains both content and metadata
    stored_object = StoredObject(content=obj.model_dump(), metadata=metadata or {})

    obj_bytes = stored_object.model_dump_json().encode("utf-8")
    # Encode key to ensure it's filesystem-friendly
    encoded_key = _encode_key(key)
    kv_store.mset([(encoded_key, obj_bytes)])
    logger.debug(f"add key '{class_name}/{encoded_key}' to kv_store {kv_store}'")


def load_object_from_kvstore(model_class: type[T], key: str | dict, kv_store_id: str = "file") -> StoredObject | None:
    """Read a Pydantic object from a key-value store.

    Args:
        model_class: Pydantic model class to reconstruct.
        key: Unique identifier for the stored object (str or dict). If dict, its hash is used.
        kv_store_id: Identifier for the storage backend.

    Returns:
        StoredObject containing the content and metadata, or None if not found.
    """
    # Use the lowercase class name of the Pydantic model
    class_name = model_class.__name__
    kv_store = KvStoreFactory(id=kv_store_id, root=class_name).get()
    # Encode key to ensure it's filesystem-friendly
    encoded_key = _encode_key(key)

    stored_bytes = kv_store.mget([encoded_key])[0]
    if not stored_bytes:
        return None
    else:
        try:
            logger.debug(f"read '{class_name}/{encoded_key}' from KV store")
            # Parse the stored object
            stored_object = StoredObject.model_validate_json(stored_bytes.decode("utf-8"))

            # Parse the content dict into the expected model class
            content = model_class.model_validate(stored_object.content)
            
            # Return a new StoredObject with the properly typed content
            return StoredObject(content=content, metadata=stored_object.metadata)
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
    with TemporaryDirectory(delete=False) as temp_dir:
        test_model = TestModel(name="test_object", value=42)
        save_object_to_kvstore("unique_key", test_model, "file", metadata={"created_by": "test", "version": "1.0"})
        retrieved_model = load_object_from_kvstore(TestModel, "unique_key", "file")
        print("File storage test:")
        if retrieved_model:
            print(f"  Content: {retrieved_model.content}")
            print(f"  Metadata: {retrieved_model.metadata}")
        else:
            print("  Not found")

    # Test Postgres SQL-based storage
    try:
        test_model_sql = TestModel(name="sql_test_object", value=123)
        save_object_to_kvstore(
            "sql_unique_key", test_model_sql, "sql", metadata={"source": "database", "priority": "high"}
        )
        retrieved_model_sql = load_object_from_kvstore(TestModel, "sql_unique_key", "sql")
        print("SQL storage test:")
        if retrieved_model_sql:
            print(f"  Content: {retrieved_model_sql.content}")
            print(f"  Metadata: {retrieved_model_sql.metadata}")
        else:
            print("  Not found")
    except Exception as e:
        print(f"SQL storage test failed (expected if SQL not configured): {e}")

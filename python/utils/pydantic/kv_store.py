"""Local file-based storage for Pydantic objects.

Provides functions to save and retrieve Pydantic object using a key-value store.
Keys are automatically encoded for filesystem compatibility.
Supports both string and dictionary keys.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Type, TypeVar

from langchain.storage import LocalFileStore
from loguru import logger
from pydantic import BaseModel, ValidationError
from unidecode import unidecode

from python.config import global_config

T = TypeVar("T", bound=BaseModel)


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


def save_pydantic_to_store(key: str | dict, obj: BaseModel, file_store_path: Path | None = None) -> None:
    """Save a Pydantic model to a local file-based key-value store.

    The model is saved to a directory based on the model's class name. The key is
    encoded to ensure filesystem compatibility.

    Args:
        key: Unique identifier for the object (str or dict). If dict, its hash is used.
        obj: Pydantic model instance to save.
        file_store_path: Root directory for file storage.
    """
    # Use the lowercase class name of the Pydantic model if no config_key is provided
    class_name = obj.__class__.__name__

    if file_store_path is None:
        file_store_path = Path(global_config().get_str("kv_store", "path"))

    dir_root_name = file_store_path / class_name
    file_store = LocalFileStore(dir_root_name)

    obj_bytes = obj.model_dump_json().encode("utf-8")
    # Encode key to ensure it's filesystem-friendly
    encoded_key = _encode_key(key)
    file_store.mset([(encoded_key, obj_bytes)])
    logger.debug(f"add key '{class_name}/{encoded_key}' to local kv_store {file_store_path}'")


def read_pydantic_from_store(model_class: type[T], key: str | dict, file_store_path: Path | None = None) -> T | None:
    """Read a Pydantic model from a local file-based key-value store.

    Args:
        model_class: Pydantic model class to reconstruct.
        key: Unique identifier for the stored object (str or dict). If dict, its hash is used.
        file_store_path: Root directory for file storage. If None, use key "kv_store", "path" in global config.

    Returns:
        Instance of the specified Pydantic model, or None if not found.
    """
    # Use the lowercase class name of the Pydantic model

    if file_store_path is None:
        file_store_path = Path(global_config().get_str("kv_store", "path"))
    class_name = model_class.__name__
    file_store = LocalFileStore(file_store_path / class_name)

    # Encode key to ensure it's filesystem-friendly
    encoded_key = _encode_key(key)

    stored_bytes = file_store.mget([encoded_key])[0]
    if not stored_bytes:
        return None
    else:
        try:
            logger.debug(f"read '{class_name}/{encoded_key}' from KV store")
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

    with TemporaryDirectory(delete=False) as temp_dir:
        temp_path = Path(temp_dir)
        test_model = TestModel(name="test_object", value=42)
        save_pydantic_to_store("unique_key", test_model, temp_path)
        retrieved_model = read_pydantic_from_store(TestModel, "unique_key", temp_path)
        print(retrieved_model)

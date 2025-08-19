"""Factory for creating and managing key-value stores with LangChain storage backends.

Provides a factory pattern for creating ByteStore instances with support for
different storage backends like local file storage and PostgreSQL.

WORK IN PROGRESS
"""

import hashlib
import json
import re
from typing import TypeVar

from langchain.storage import LocalFileStore
from langchain_core.stores import ByteStore
from loguru import logger
from pydantic import BaseModel, ValidationError
from unidecode import unidecode

from src.utils.config_mngr import global_config

kvstore_config = global_config().get_dict("kv_store")


class KvStoreFactory(BaseModel):
    """Factory for creating key-value stores with configurable backends.

    Attributes:
        id: Identifier for the storage backend type
        root: Root namespace or directory for the storage
    """

    id: str
    root: str = ""

    def get(self) -> ByteStore:
        """Create and return a ByteStore instance based on the configured backend.

        Returns:
            ByteStore: A configured ByteStore instance
        """
        if self.id == "file":
            path = global_config().get_dir_path(f"kv_store.{self.id}.path")
            return LocalFileStore(path / self.root)
        elif self.id == "sql":
            from langchain_community.storage import SQLStore

            db_url = global_config().get_dsn(f"kv_store.{self.id}.path", driver=None)  #  async not supported yet
            print(db_url)
            store = SQLStore(namespace=self.root, db_url=db_url)
            store.create_schema()
            return SQLStore(namespace=self.root, db_url=db_url)

        else:
            raise ValueError(f"Unknown vector store id: '{self.id}'")


T = TypeVar("T", bound=BaseModel)


class StoredObject(BaseModel):
    """Wrapper for storing Pydantic objects with metadata."""

    content: dict
    metadata: dict = {}

    @classmethod
    def from_model(cls, model: BaseModel, metadata: dict | None = None) -> "StoredObject":
        """Create StoredObject from a Pydantic model."""
        content = model.model_dump()
        fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
        merged_metadata = {"fingerprint": fingerprint, **(metadata or {})}
        return cls(content=content, metadata=merged_metadata)

    @classmethod
    def parse_model(cls, data: dict, model_class: type[T]) -> "StoredObject":
        """Parse stored data using a specific model class."""
        content_data = data.get("content", {})
        return cls(content=content_data, metadata=data.get("metadata", {}))

    def to_model(self, model_class: type[T]) -> T:
        """Convert stored content back to a Pydantic model."""
        return model_class.model_validate(self.content)


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


class PydanticStoreFactory(BaseModel):
    """Factory for storing and retrieving Pydantic objects from key-value stores."""
    
    id: str
    model_class: type[T]

    def _get_kv_store(self) -> ByteStore:
        """Get the underlying ByteStore instance."""
        class_name = self.model_class.__name__
        return KvStoreFactory(id=self.id, root=class_name).get()

    def save_obj(self, key: str | dict, obj: BaseModel, metadata: dict | None = None) -> None:
        """Save a Pydantic model to the key-value store.
        
        Args:
            key: Unique identifier for the object (str or dict). If dict, its hash is used.
            obj: Pydantic model instance to save.
            metadata: Optional dictionary of metadata to store with the object
        """
        kv_store = self._get_kv_store()
        stored_obj = StoredObject.from_model(obj, metadata)
        obj_bytes = stored_obj.model_dump_json().encode("utf-8")
        encoded_key = _encode_key(key)
        kv_store.mset([(encoded_key, obj_bytes)])
        logger.debug(f"add key '{self.model_class.__name__}/{encoded_key}' to kv_store {kv_store}'")

    def load_object(self, key: str | dict) -> T | None:
        """Read a Pydantic object from the key-value store.
        
        Args:
            key: Unique identifier for the stored object (str or dict). If dict, its hash is used.

        Returns:
            Instance of model_class with metadata attached as '_metadata' attribute, or None if not found.
        """
        kv_store = self._get_kv_store()
        encoded_key = _encode_key(key)
        
        stored_bytes = kv_store.mget([encoded_key])[0]
        if not stored_bytes:
            return None
            
        try:
            logger.debug(f"read '{self.model_class.__name__}/{encoded_key}' from KV store")
            stored_data = json.loads(stored_bytes.decode("utf-8"))
            logger.debug(f"stored_data: {stored_data}")

            # Try loading as StoredObject format
            try:
                stored_obj = StoredObject.parse_model(stored_data, self.model_class)
                # Convert back to the original model and attach metadata
                model_instance = stored_obj.to_model(self.model_class)
                # Attach metadata as an attribute
                setattr(model_instance, "_metadata", stored_obj.metadata)  # noqa: B010
                return model_instance
            except (ValidationError, KeyError):
                logger.debug("Failed to load as StoredObject, trying legacy format...")
                # Try loading as legacy direct object
                model_instance = self.model_class.model_validate(stored_data)
                # Attach empty metadata for legacy objects
                setattr(model_instance, "_metadata", {"legacy": True})  # noqa: B010
                return model_instance

        except ValidationError as ex:
            logger.warning(f"failed to load JSON value for {self.model_class.__name__}/{encoded_key}. Error is : {ex}")
            return None
        except Exception as ex:
            logger.warning(f"failed to load JSON value for {self.model_class.__name__}/{encoded_key}. Exception is : {ex}")
            return None

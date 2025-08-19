"""Key_value storage for Pydantic objects.

Provides functions to save and retrieve Pydantic object using a key-value store.
Keys are automatically encoded for filesystem compatibility.
Supports both string and dictionary keys.
"""

from pydantic import BaseModel

from src.ai_extra.kv_store_factory import PydanticStoreFactory

T = TypeVar("T", bound=BaseModel)


def save_object_to_kvstore(
    key: str | dict, obj: BaseModel, metadata: dict | None = None, kv_store_id: str = "file"
) -> None:
    """Save a Pydantic model to a local file-based key-value store.

    The model is saved to a directory based on the model's class name. The key is
    encoded to ensure filesystem compatibility.

    Args:
        key: Unique identifier for the object (str or dict). If dict, its hash is used.
        obj: Pydantic model instance to save.
        metadata: Optional dictionary of metadata to store with the object
        kv_store_id: Identifier for the key-value store backend
    """
    factory = PydanticStoreFactory(id=kv_store_id, model_class=obj.__class__)
    factory.save_obj(key, obj, metadata)


def load_object_from_kvstore(model_class: type[T], key: str | dict, kv_store_id: str = "file") -> T | None:
    """Read a Pydantic object and its metadata from a key-value store.

    Args:
        model_class: Pydantic model class to reconstruct.
        key: Unique identifier for the stored object (str or dict). If dict, its hash is used.
        kv_store_id: Identifier for the key-value store backend

    Returns:
        Instance of model_class with metadata attached as '_metadata' attribute, or None if not found.
    """
    factory = PydanticStoreFactory(id=kv_store_id, model_class=model_class)
    return factory.load_object(key)


# Quick test
if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from devtools import debug

    class TestModel(BaseModel):
        name: str
        value: int

    # Test file-based storage
    with TemporaryDirectory(delete=False) as temp_dir:
        test_model = TestModel(name="test_object", value=42)
        save_object_to_kvstore("unique_key", test_model, kv_store_id="file", metadata={"some_metadata": 55})
        retrieved_model = load_object_from_kvstore(TestModel, "unique_key", kv_store_id="file")
        debug("File storage test:", retrieved_model)
        if retrieved_model and hasattr(retrieved_model, "_metadata"):
            debug("File storage metadata:", retrieved_model._metadata)

    # Test Postgres SQL-based storage
    try:
        test_model_sql = TestModel(name="sql_test_object", value=123)
        save_object_to_kvstore("sql_unique_key", test_model_sql, kv_store_id="sql")
        retrieved_model_sql = load_object_from_kvstore(TestModel, "sql_unique_key", kv_store_id="sql")
        debug("SQL storage test:", retrieved_model_sql)
        if retrieved_model_sql and hasattr(retrieved_model_sql, "_metadata"):
            debug("SQL storage metadata:", retrieved_model_sql.__getattribute__("_metadata"))
    except Exception as e:
        print(f"SQL storage test failed (expected if SQL not configured): {e}")

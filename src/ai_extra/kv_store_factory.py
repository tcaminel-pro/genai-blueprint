"""Factory for creating and managing key-value stores with LangChain storage backends.

Provides a factory pattern for creating ByteStore instances with support for
different storage backends like local file storage and PostgreSQL.

WORK IN PROGRESS
"""

from langchain.storage import LocalFileStore
from langchain_core.stores import ByteStore
from pydantic import BaseModel

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

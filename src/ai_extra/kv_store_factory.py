#  Add doc AI!
from langchain.storage import LocalFileStore
from langchain_core.stores import ByteStore
from pydantic import BaseModel

from src.utils.config_mngr import global_config

# KV_STORE_TYPE = Literal["file", "Chroma_in_memory", "InMemory", "Sklearn", "PgVector"]

kvstore_config = global_config().get_dict("kv_store")


class KvStoreFactory(BaseModel):
    id: str
    root: str = ""

    def get(self) -> ByteStore:
        if self.id == "file":
            path = global_config().get_dir_path(f"kv_store.{self.id}.path")
            return LocalFileStore(path / self.root)
        elif self.id == "postgres":
            # NOT TESTED
            from langchain_community.storage import SQLStore

            db_url = global_config().get_str(f"kv_store.{self.id}.postgres_url")
            return SQLStore(namespace=self.root, db_url=db_url)

        else:
            raise ValueError(f"Unknown vector store id: '{self.id}'")

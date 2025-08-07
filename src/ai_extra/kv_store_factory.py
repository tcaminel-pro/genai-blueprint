from langchain.storage import LocalFileStore
from langchain_community.storage import SQLStore
from langchain_core.stores import ByteStore


class KvStoreFactory(BaseModel):
    def get() -> ByteStore:
        if 1:
            return LocalFileStore(...)
        if 2:
            return SQLStore(...)

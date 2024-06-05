"""Vector store factory and configuration


"""

from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal, get_args

from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import ConfigurableField, Runnable
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.ai_core.embeddings import EmbeddingsFactory
from python.config import get_config

# from langchain_chroma import Chroma  does not work (yet?) with self_query


default_collection = get_config("vector_store", "default_collection")


def get_vector_vector_store_path() -> str:
    # get path to store vector database, as specified in configuration file
    dir = Path(get_config("vector_store", "path"))
    try:
        dir.mkdir()
    except Exception:
        pass  # TODO : log something
    return str(dir)


VectorStoreLiterals = Literal["Chroma", "Chroma_in_memory"]


class VectorStoreFactory(BaseModel):
    id: Annotated[VectorStoreLiterals | None, Field(validate_default=True)] = None
    # id_: Literal["Chroma", "Chroma_in_memory"] | None = None
    embeddings_factory: EmbeddingsFactory
    collection_name: str = default_collection
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    _record_manager: SQLRecordManager | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def collection_full_name(self) -> str:
        assert self.embeddings_factory
        embeddings_id = self.embeddings_factory.info.id
        return f"{self.collection_name}_{embeddings_id}"

    @computed_field
    def description(self) -> str:
        r = f"{str(self.id)}/{self.collection_full_name}"
        if self.id == "Chroma":
            r += f" => store: {get_vector_vector_store_path()}"
        if self.index_document == Chroma and self._record_manager:
            r += f" indexer: {self._record_manager}"
        return r

    @staticmethod
    def known_items() -> list[str]:
        return list(get_args(VectorStoreLiterals))

    @field_validator("id", mode="before")
    def check_known(cls, id: str) -> str:
        if id is None:
            id = get_config("vector_store", "default")
        if id not in VectorStoreFactory.known_items():
            raise ValueError(f"Unknown Vector Store: {id}")
        return id

    @computed_field
    @cached_property
    def vector_store(self) -> VectorStore:
        """
        Factory for the vector database
        """
        embeddings = self.embeddings_factory.get()

        if self.id == "Chroma":
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=get_vector_vector_store_path(),
                collection_name=self.collection_full_name,
                collection_metadata=self.collection_metadata,
            )
        elif self.id == "Chroma_in_memory":
            vector_store = Chroma(
                embedding_function=embeddings,
                collection_name=self.collection_full_name,
                collection_metadata=self.collection_metadata,
            )
        else:
            raise ValueError(f"Unknown vector store: {self.id}")

        logger.info(f"get vector store  : {self.description}")
        if self.index_document:
            db_url = (
                f"sqlite:///{get_vector_vector_store_path()}/record_manager_cache.sql"
            )
            logger.info(f"vector store record manager : {db_url}")
            namespace = f"{id}/{self.collection_full_name}"
            self._record_manager = SQLRecordManager(
                namespace,
                db_url=db_url,  # @TODO: To improve !!
            )
            self._record_manager.create_schema()

        return vector_store

    def get_configurable_retriever(self, default_k: int = 4) -> Runnable:
        """
        Return a retriever where we can configure k at run-time.

        Should be called as:
            config = {"configurable": {"search_kwargs": {"k": count}}}
            result = retriever.invoke(user_input, config=config)
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": default_k}
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
            )
        )
        return retriever

    def add_documents(self, docs: Iterable[Document]) -> IndexingResult | list[str]:
        # TODO : accept BaseLoader
        if not self.index_document:
            return self.vector_store.add_documents(list(docs))
        else:
            vector_store = self.vector_store
            assert self._record_manager

            info = index(
                docs,
                self._record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
            )
            return info

    def document_count(self):
        # It seems there's no generic way to get the number of docs stored in a Vector Store.
        if self.id in ["Chroma", "Chroma_in_memory"]:
            return self.vector_store._collection.count()  # type: ignore
        else:
            raise NotImplementedError(
                f"Don'k know how to get collection count for {self.vector_store}"
            )


def search_one(vc: VectorStore, query: str):
    return vc.similarity_search(query, k=1)

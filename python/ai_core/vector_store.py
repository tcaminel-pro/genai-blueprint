"""Vector store factory and configuration


"""

from functools import cache
from pathlib import Path

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores import Chroma

from python.ai_core.embeddings import embeddings_factory
from python.config import get_config

# from langchain_chroma import Chroma  does not work (yet?) with self_query


default_collection = get_config("vector_store", "default_collection")


def get_vector_vector_store_path() -> str:
    # get path to store vector database, as specified in configuration file
    dir = Path(get_config("vector_store", "path"))
    try:
        dir.mkdir()
    except:
        pass  # TODO : log something
    return str(dir)


def vector_store_factory(
    name: str | None = None,
    embeddings: Embeddings | None = None,
    collection_name: str = default_collection,
) -> VectorStore:
    """
    Factory for the vector database
    """
    if name is None:
        name = get_config("vector_store", "default")
    if embeddings is None:
        embeddings = embeddings_factory()  # Get default one
    if name == "Chroma":
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=get_vector_vector_store_path(),
            collection_name=collection_name,
        )
    elif name == "Chroma_in_memory":
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"Unknown vector store: {name}")
    return vector_store


def search_one(vc: VectorStore, query: str):
    return vc.similarity_search(query, k=1)


def document_count(vs: VectorStore):
    # It seems there's no generic way to get the number of docs stored in a Vector Store.
    if isinstance(vs, Chroma):
        return vs._collection.count()
    else:
        raise NotImplemented(f"Don'k know how to get collection count for {vs}")

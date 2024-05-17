"""Vector store factory and configuration


"""

from pathlib import Path

from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.chroma import Chroma

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


def vector_store_factory(
    id: str | None = None,
    embeddings_factory: EmbeddingsFactory | None = None,
    collection_name: str = default_collection,
) -> VectorStore:
    """
    Factory for the vector database
    """
    if id is None:
        id = get_config("vector_store", "default")
    if embeddings_factory is None:
        embeddings_factory = EmbeddingsFactory()  # Get default one
    embeddings = embeddings_factory.get()
    embeddings_id = embeddings_factory.info.id
    collection_name = f"{collection_name}_{embeddings_id}"
    if id == "Chroma":
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=get_vector_vector_store_path(),
            collection_name=collection_name,
        )
    elif id == "Chroma_in_memory":
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"Unknown vector store: {id}")
    return vector_store


def search_one(vc: VectorStore, query: str):
    return vc.similarity_search(query, k=1)


def document_count(vs: VectorStore):
    # It seems there's no generic way to get the number of docs stored in a Vector Store.
    if isinstance(vs, Chroma):
        return vs._collection.count()
    else:
        raise NotImplementedError(f"Don'k know how to get collection count for {vs}")

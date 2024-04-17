"""Vector store factory and configuration


"""

from functools import cache
from pathlib import Path


from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma

from python.config import get_config


DEFAULT_COLLECTION = "Training_collection"


def get_vector_vector_store_path() -> str:
    # get path to store vector database, as specified in configuration file
    dir = Path(get_config("embeddings", "vector_store_path"))
    try:
        dir.mkdir()
    except:
        pass
    return str(dir)


def get_vector_store(
    name: str = "Chroma",
    embeddings: Embeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStore:
    # Singleton for the vector database object (ChromaDB)
    if name == "Chroma":
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=get_vector_vector_store_path(),
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"Unknown vector store: {name}")
    return vector_store

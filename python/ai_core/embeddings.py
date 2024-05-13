"""
Embedding models factory.
They can  be either Cloud based or for local run with CPU

"""

# See also https://huggingface.co/spaces/mteb/leaderboard

from functools import cache

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from python.config import get_config


@cache
def embeddings_factory(embeddings_model: str | None = None) -> Embeddings:
    """
    Return an embedding function by name.  is none is given, take the default defined in configuration file.
    """

    if embeddings_model is None:
        embeddings_model = get_config("embeddings", "default_model")

    if not embeddings_model:
        embeddings_model = "OpenAI"

    match embeddings_model.partition("/")[0]:
        case "OpenAI":
            return OpenAIEmbeddings()

        case "sentence-transformers" | "dangvantuan":
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

            cache = get_config("embeddings", "cache")

            return HuggingFaceEmbeddings(
                model_name=embeddings_model,
                model_kwargs={"device": "cpu"},
                cache_folder=cache,
            )

        case _:
            raise ValueError("Unknown embedding model", embeddings_model)

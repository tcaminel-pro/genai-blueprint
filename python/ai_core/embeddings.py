"""
Embedding models factory.
They can  be either Cloud based or for local run with CPU

"""

# See also https://huggingface.co/spaces/mteb/leaderboard

import os
from functools import cache, cached_property

from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.config import get_config_str


class EMBEDDINGS_INFO(BaseModel):
    id: str  # a given ID for the embeddings
    cls: str  # Name of the constructor
    model: str  # Provider name of the model
    key: str  # API key
    prefix: str = ""  # Some LLM requires a prefix in the call.  To be improved.

    def get_key(self):
        key = os.environ.get(self.key)
        if key is None:
            raise ValueError(f"No environment variable for {self.key} ")
        return key

    def __hash__(self):
        return hash(self.id)


KNOWN_EMBEDDINGS_MODELS = [
    EMBEDDINGS_INFO(
        id="ada_002_openai",
        model="text-embedding-ada-002",
        cls="OpenAIEmbeddings",
        key="OPENAI_API_KEY",
    ),
    EMBEDDINGS_INFO(
        id="embedding_001_google",
        model="text-embedding-ada-002",
        cls="GoogleGenerativeAIEmbeddings",
        key="GOOGLE_API_KEY",
    ),
    EMBEDDINGS_INFO(
        id="multilingual_MiniLM_local",
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cls="HuggingFaceEmbeddings",
        key="",
    ),
    EMBEDDINGS_INFO(
        id="ada_002_edenai",
        model="openai/1536__text-embedding-ada-002",
        cls="EdenAiEmbeddings",
        key="EDENAI_API_KEY",
    ),
    EMBEDDINGS_INFO(
        id="mistral_1024_edenai",
        model="mistral/1024__mistral-embed",
        cls="EdenAiEmbeddings",
        key="EDENAI_API_KEY",
    ),
    EMBEDDINGS_INFO(
        id="camembert_large_local",
        model="dangvantuan/sentence-camembert-large",
        cls="HuggingFaceEmbeddings",
        key="",
    ),
    EMBEDDINGS_INFO(
        id="solon-large",
        model="OrdalieTech/Solon-embeddings-large-0.1",
        cls="HuggingFaceEmbeddings",
        key="",
        prefix="Query : ",
    ),
]


class EmbeddingsFactory(BaseModel):
    embeddings_id: Annotated[str | None, Field(validate_default=True)] = None
    encoding_str: str | None = None
    retrieving_str: str | None = None

    @computed_field
    @cached_property
    def info(self) -> EMBEDDINGS_INFO:
        assert self.embeddings_id
        return EmbeddingsFactory.known_items_dict().get(self.embeddings_id)  # type: ignore

    @field_validator("embeddings_id", mode="before")
    @classmethod
    def check_known(cls, embeddings_id: str) -> str:
        if embeddings_id is None:
            embeddings_id = get_config_str("embeddings", "default_model")
        if embeddings_id not in EmbeddingsFactory.known_items():
            raise ValueError(f"Unknown Embeddings: {embeddings_id}")
        return embeddings_id

    @staticmethod
    def known_items_dict() -> dict[str, EMBEDDINGS_INFO]:
        return {
            item.id: item
            for item in KNOWN_EMBEDDINGS_MODELS
            if item.key in os.environ or item.key == ""
        }

    @staticmethod
    def known_items() -> list[str]:
        return list(EmbeddingsFactory.known_items_dict().keys())

    def get(self) -> Embeddings:
        """
        Create an embeddings model object.

        """
        if self.info.key not in os.environ and self.info.key != "":
            raise ValueError(f"No known API key for : {self.info.id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> Embeddings:
        if self.info.cls == "OpenAIEmbeddings":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings()
        elif self.info.cls == "GoogleGenerativeAIEmbeddings":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            emb = GoogleGenerativeAIEmbeddings(model=self.info.model)  # type: ignore
        elif self.info.cls == "HuggingFaceEmbeddings":
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

            cache = get_config_str("embeddings", "cache")
            emb = HuggingFaceEmbeddings(
                model_name=self.info.model,
                model_kwargs={"device": "cpu"},
                cache_folder=cache,
            )
        elif self.info.cls == "EdenAiEmbeddings":
            from langchain_community.embeddings.edenai import EdenAiEmbeddings

            provider, _, model = self.info.model.partition("/")
            emb = EdenAiEmbeddings(model=model, provider=provider, edenai_api_key=None)
        else:
            raise ValueError(f"unsupported Embeddings class {self.info.cls}")
        return emb


@cache
def get_embeddings(
    embeddings_id: str | None = None,
    encoding_str: str | None = None,
    retrieving_str: str | None = None,
) -> Embeddings:
    """
    Get an embeddings model.
    - embeddings_id is its id.  If None, take the model defined in configuration
    - encoding_str, retrieving_str : not used yet
    """
    factory = EmbeddingsFactory(
        embeddings_id=embeddings_id,
        encoding_str=encoding_str,
        retrieving_str=retrieving_str,
    )
    return factory.get()

"""
Embedding models factory.
They can  be either Cloud based or for local run with CPU

"""

# See also https://huggingface.co/spaces/mteb/leaderboard

import os
from functools import cache, cached_property, lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.config import get_config_str

_ = load_dotenv(verbose=True)


class EmbeddingsInfo(BaseModel):
    id: str  # a given ID for the embeddings
    cls: str  # Name of the constructor
    model: str  # Provider name of the model
    key: str | None = None  # API key
    prefix: str = ""  # Some LLM requires a prefix in the call.  To be improved.

    def get_key(self):
        if self.key:
            key = os.environ.get(self.key)
            if key is None:
                raise ValueError(f"No environment variable for {self.key} ")
            return key
        else:
            return ""

    def __hash__(self):
        return hash(self.id)


def read_embeddings_list_file() -> list[EmbeddingsInfo]:
    yml_file = Path(get_config_str("embeddings", "list"))
    assert yml_file.exists(), f"cannot find {yml_file}"
    with open(yml_file, "r") as f:
        data = yaml.safe_load(f)
    embedding = [EmbeddingsInfo(**e) for e in data["embeddings"]]
    return embedding


class EmbeddingsFactory(BaseModel):
    embeddings_id: Annotated[str | None, Field(validate_default=True)] = None
    encoding_str: str | None = None
    retrieving_str: str | None = None

    @computed_field
    @cached_property
    def info(self) -> EmbeddingsInfo:
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

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[EmbeddingsInfo]:
        return read_embeddings_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, EmbeddingsInfo]:
        return {
            item.id: item
            for item in EmbeddingsFactory.known_list()
            if item.key is None or item.key in os.environ
        }

    @staticmethod
    def known_items() -> list[str]:
        return list(EmbeddingsFactory.known_items_dict().keys())

    def get(self) -> Embeddings:
        """
        Create an embeddings model object.

        """
        if self.info.key and self.info.key not in os.environ:
            raise ValueError(f"No known API key for : {self.info.id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> Embeddings:
        if self.info.cls == "OpenAIEmbeddings":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings()
        elif self.info.cls == "GoogleGenerativeAIEmbeddings":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore  # noqa: I001

            emb = GoogleGenerativeAIEmbeddings(model=self.info.model)  # type: ignore
        elif self.info.cls == "HuggingFaceEmbeddings":
            from langchain_huggingface import HuggingFaceEmbeddings

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
        elif self.info.cls == "AzureOpenAIEmbeddings":
            from langchain_openai import AzureOpenAIEmbeddings

            name, _, api_version = self.info.model.partition("/")
            emb = AzureOpenAIEmbeddings(
                azure_deployment=name,
                model=name,  # Not sure it's needed
                api_version=api_version,
            )
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

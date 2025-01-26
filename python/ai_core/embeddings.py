"""Embedding models factory and management.

This module provides a factory pattern implementation for creating and managing
embedding models from various providers. It supports both cloud-based and local
CPU-based models.

Key Features:
- Support for multiple embedding providers (OpenAI, HuggingFace, Google, etc.)
- Configuration through YAML files
- Automatic API key management
- Model caching and persistence
- Integration with vector stores

Supported Providers:
- OpenAI
- Google Generative AI
- HuggingFace
- EdenAI
- Azure OpenAI
- DeepSeek
- Ollama

Example:
     # Get default embeddings
     embeddings = get_embeddings()

     # Get specific model
     embeddings = get_embeddings(embeddings_id="huggingface_all-mpnet-base-v2")

"""

import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from python.config import global_config

_ = load_dotenv(verbose=True)


class EmbeddingsInfo(BaseModel):
    """Information about an embeddings model.

    Attributes:
        id (str): A unique identifier for the embeddings model.
        cls (str): The name of the constructor for the embeddings model.
        model (str): The provider name of the model.
        key (str | None): The API key for the model, if required.
        prefix (str): A prefix required by some models in the API call.

    Example:
         info = EmbeddingsInfo(
        ...     id="example_model",
        ...     cls="ExampleEmbeddings",
        ...     model="example_provider",
        ...     key="API_KEY_ENV_VAR"
        ... )
         info.get_key()  # Retrieves API key from environment
    """

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


def _read_embeddings_list_file() -> list[EmbeddingsInfo]:
    yml_file = Path(global_config().get_str("embeddings", "list"))
    assert yml_file.exists(), f"cannot find {yml_file}"
    with open(yml_file) as f:
        data = yaml.safe_load(f)
    embeddings = []
    for provider in data["embeddings"]:
        cls = provider["cls"]
        for model in provider["models"]:
            model["cls"] = cls
            embeddings.append(EmbeddingsInfo(**model))
    return embeddings


class EmbeddingsFactory(BaseModel):
    """Factory class for creating and managing embeddings models.

    Attributes:
        embeddings_id (str | None): The unique identifier for the embeddings model.
        encoding_str (str | None): The encoding string for the model.
        retrieving_str (str | None): The retrieving string for the model.

    Example:
    ```
         # Get default embeddings factory
         factory = EmbeddingsFactory()

         # Get specific model
         factory = EmbeddingsFactory(embeddings_id="multilingual_MiniLM_local")
         embeddings = factory.get()
         vectors = embeddings.embed_documents(["Sample text"])
    """

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
            embeddings_id = global_config().get_str("embeddings", "default_model")
        if embeddings_id not in EmbeddingsFactory.known_items():
            raise ValueError(f"Unknown Embeddings: {embeddings_id}")
        return embeddings_id

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[EmbeddingsInfo]:
        return _read_embeddings_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, EmbeddingsInfo]:
        return {item.id: item for item in EmbeddingsFactory.known_list() if item.key is None or item.key in os.environ}

    @staticmethod
    def known_items() -> list[str]:
        return list(EmbeddingsFactory.known_items_dict().keys())

    def get(self) -> Embeddings:
        """Create an embeddings model object."""
        if self.info.key and self.info.key not in os.environ:
            raise ValueError(f"No known API key for : {self.info.id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> Embeddings:
        """Create an embeddings model object based on the configuration.

        Returns:
            Embeddings: An instance of the configured embeddings model.
        """
        if self.info.cls == "OpenAIEmbeddings":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings()
        elif self.info.cls == "GoogleGenerativeAIEmbeddings":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore  # noqa: I001

            emb = GoogleGenerativeAIEmbeddings(model=self.info.model)  # type: ignore
        elif self.info.cls == "HuggingFaceEmbeddings":
            from langchain_huggingface import HuggingFaceEmbeddings

            cache = global_config().get_str("embeddings", "cache")
            emb = HuggingFaceEmbeddings(
                model_name=self.info.model,
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
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
        # elif self.info.cls == "DeepSeekCoderEmbeddings":
        #     from langchain_deepseek import DeepSeekCoderEmbeddings

        #     emb = DeepSeekCoderEmbeddings(model=self.info.model)
        elif self.info.cls == "OllamaEmbeddings":
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(model=self.info.model)

        else:
            raise ValueError(f"unsupported Embeddings class {self.info.cls}")
        return emb


def get_embeddings(
    embeddings_id: str | None = None,
    encoding_str: str | None = None,
    retrieving_str: str | None = None,
) -> Embeddings:
    """Get an embeddings model.

    Args:
        embeddings_id: The unique identifier for the embeddings model. If None, uses default from config.
        encoding_str: Optional encoding string (currently unused).
        retrieving_str: Optional retrieving string (currently unused).

    Returns:
        Embeddings: An instance of the configured embeddings model.

    Example:
    ```
         # Get default embeddings
         embeddings = get_embeddings()

         # Get specific model
         embeddings = get_embeddings(embeddings_id="huggingface_all-mpnet-base-v2")
         vectors = embeddings.embed_documents(["Sample text"])
    """
    factory = EmbeddingsFactory(
        embeddings_id=embeddings_id,
        encoding_str=encoding_str,
        retrieving_str=retrieving_str,
    )
    logger.info(f"get logger: '{factory.embeddings_id}'")
    return factory.get()

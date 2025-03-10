"""Embedding models factory and management.

This module provides a comprehensive factory pattern implementation for creating
and managing embedding models from various providers. It supports a wide range
of embedding technologies across cloud-based and local CPU-based models.

Key Features:
- Unified interface for multiple embedding providers
- Dynamic configuration through configuration files
- Secure API key management
- Flexible model caching and persistence
- Seamless integration with vector stores and machine learning workflows

Supported Providers:
- OpenAI
- Google Generative AI
- HuggingFace
- EdenAI
- Azure OpenAI
- Ollama

Example:
    # Get default embeddings
    embeddings = get_embeddings()

    # Get specific model
    embeddings = get_embeddings(embeddings_id="huggingface_all-mpnet-base-v2")
    vectors = embeddings.embed_documents(["Sample text"])
"""

import os
from functools import cached_property, lru_cache
from typing import Annotated

import yaml
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.storage import LocalFileStore
from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from src.utils.config_mngr import global_config

_ = load_dotenv(verbose=True)


class EmbeddingsInfo(BaseModel):
    """Information about an embeddings model.

    Provides comprehensive details about an embedding model, including
    its unique identifier, constructor, provider, and optional API key.

    Attributes:
        id: Unique identifier for the embeddings model
        cls: Name of the embeddings model constructor
        model: Provider name of the model
        key: Optional API key for accessing the model
        prefix: Optional prefix required by some models in API calls
    """

    id: str
    provider: str
    model: str
    key: str | None = None
    prefix: str = ""

    def get_key(self):
        """Retrieve the API key from environment variables.

        Returns:
            API key string or empty string if no key is required

        Raises:
            ValueError: If configured key is not found in environment
        """
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
    """Read embeddings configuration from YAML file.

    Returns:
        List of configured embeddings models

    Raises:
        AssertionError: If configuration file is not found
    """
    yml_file = global_config().get_path("embeddings.list")
    with open(yml_file) as f:
        data = yaml.safe_load(f)
    embeddings = []
    for provider in data["embeddings"]:
        cls = provider["provider"]
        for model in provider["models"]:
            model["provider"] = cls
            embeddings.append(EmbeddingsInfo(**model))
    return embeddings


class EmbeddingsFactory(BaseModel):
    """Factory for creating and managing embeddings models.

    Provides a flexible and configurable way to instantiate embedding models
    from various providers with support for caching and dynamic configuration.

    Attributes:
        embeddings_id: Unique identifier for the embeddings model
        encoding_str: Optional encoding configuration string
        retrieving_str: Optional retrieving configuration string
    """

    embeddings_id: Annotated[str | None, Field(validate_default=True)] = None
    encoding_str: str | None = None
    retrieving_str: str | None = None

    @computed_field
    @cached_property
    def info(self) -> EmbeddingsInfo:
        """Retrieve embeddings model information.

        Returns:
            Configuration details for the selected embeddings model
        """
        assert self.embeddings_id
        return EmbeddingsFactory.known_items_dict().get(self.embeddings_id)  # type: ignore

    @field_validator("embeddings_id", mode="before")
    @classmethod
    def check_known(cls, embeddings_id: str) -> str:
        """Validate and normalize embeddings model identifier.
        Args:
            embeddings_id: Model identifier to validate
        Returns:
            Validated model identifier

        """
        if embeddings_id is None:
            embeddings_id = global_config().get_str("embeddings.default_model")
        if embeddings_id not in EmbeddingsFactory.known_items():
            raise ValueError(f"Unknown Embeddings: {embeddings_id}")
        return embeddings_id

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[EmbeddingsInfo]:
        """List all known embeddings models.

        Returns:
            List of all configured embeddings models
        """
        return _read_embeddings_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, EmbeddingsInfo]:
        """Create a dictionary of available embeddings models.

        Returns:
            Dictionary mapping model IDs to their configurations
        """
        return {item.id: item for item in EmbeddingsFactory.known_list() if item.key is None or item.key in os.environ}

    @staticmethod
    def known_items() -> list[str]:
        """List identifiers of available embeddings models.

        Returns:
            List of model identifiers
        """
        return list(EmbeddingsFactory.known_items_dict().keys())

    def get(self, cached: bool = False) -> Embeddings:
        """Create an embeddings model instance.

        Args:
            cached: Whether to use cached embeddings

        Returns:
            Configured embeddings model

        Raises:
            ValueError: If API key is required but not found
        """
        if self.info.key and self.info.key not in os.environ:
            raise ValueError(f"No known API key for : {self.info.id}")
        embeddings = self.model_factory()
        if cached:
            embeddings = self.get_cached_embedder()
        return embeddings

    def model_factory(self) -> Embeddings:
        """Create an embeddings model based on configuration.

        Returns:
            Instantiated embeddings model

        Raises:
            ValueError: If embeddings model is not supported
        """
        if self.info.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings()
        elif self.info.provider == "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore  # noqa: I001

            emb = GoogleGenerativeAIEmbeddings(model=self.info.model)  # type: ignore
        elif self.info.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            cache = global_config().get_str("embeddings.cache")
            emb = HuggingFaceEmbeddings(
                model_name=self.info.model,
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder=cache,
            )
        elif self.info.provider == "edenai":
            from langchain_community.embeddings.edenai import EdenAiEmbeddings

            provider, _, model = self.info.model.partition("/")
            emb = EdenAiEmbeddings(model=model, provider=provider, edenai_api_key=None)
        elif self.info.provider == "azure_openai":
            from langchain_openai import AzureOpenAIEmbeddings

            name, _, api_version = self.info.model.partition("/")
            emb = AzureOpenAIEmbeddings(
                azure_deployment=name,
                model=name,  # Not sure it's needed
                api_version=api_version,
            )
        elif self.info.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(model=self.info.model)
        elif self.info.provider == "deepinfra":
            from langchain_community.embeddings import DeepInfraEmbeddings

            emb = DeepInfraEmbeddings(model_id=self.info.model)

        else:
            raise ValueError(f"unsupported Embeddings class {self.info.provider}")
        return emb

    def get_cached_embedder(self) -> CacheBackedEmbeddings:
        """Create a cached embeddings model.

        Returns:
            Cached embeddings model with persistent storage
        """
        file_store_path = global_config().get_path("kv_store.path")
        file_store = LocalFileStore(file_store_path / "embeddings_cache")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(self.get(), file_store, namespace=self.embeddings_id)  # type: ignore
        return cached_embedder


def get_embeddings(
    embeddings_id: str | None = None,
    encoding_str: str | None = None,
    retrieving_str: str | None = None,
) -> Embeddings:
    """Retrieve an embeddings model with optional configuration.

    Provides a convenient way to get an embeddings model with flexible
    configuration options.

    Args:
        embeddings_id: Unique identifier for the embeddings model
        encoding_str: Optional encoding configuration string
        retrieving_str: Optional retrieving configuration string

    Returns:
        Configured embeddings model

    Example:
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

"""
LLM model factory and configuration management.

This module provides a factory pattern implementation for creating and configuring Language 
Learning Models (LLMs) from various providers. It supports:

- Dynamic LLM selection and configuration
- Multiple model providers (OpenAI, DeepInfra, Groq, etc.)
- Runtime model switching
- Fallback mechanisms
- Caching
- Streaming responses
- JSON mode for structured outputs

The configuration of available models is stored in models_providers.yaml.
Each model must have a unique ID following the pattern: model_version_provider
Example: gpt_35_openai for GPT-3.5 from OpenAI

The module handles API keys through environment variables defined in API_KEYS.
"""


# TODO
#  implement from langchain_core.rate_limiters import InMemoryRateLimiter


import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import cast

import yaml
from dotenv import load_dotenv
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.runnables import ConfigurableField, Runnable
from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.ai_core.cache import LlmCache, set_cache
from python.config import get_config_str

load_dotenv(verbose=True, override=True)


MAX_TOKENS = 2048


API_KEYS = {
    "ChatOpenAI": "OPENAI_API_KEY",
    "ChatDeepInfra": "DEEPINFRA_API_TOKEN",
    "ChatGroq": "GROQ_API_KEY",
    "ChatVertexAI": "GOOGLE_API_KEY",
    "ChatOllama": "",
    "ChatEdenAI": "EDENAI_API_KEY",
    "AzureChatOpenAI": "AZURE_OPENAI_API_KEY",
    "ChatTogether": "TOGETHER_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "ChatOpenrouter": "OPENROUTER_API_KEY",
}


class LlmInfo(BaseModel):
    """
    Description of an LLM model and its configuration.

    Attributes:
        id: Unique identifier in format model_version_provider (e.g. gpt_35_openai)
        cls: LangChain class name used to instantiate the model
        model: Model identifier used by the provider
        key: API key environment variable name (computed from cls)
    """

    # an ID for the LLM; should follow Python variables constraints, and have 3 parts: model_version_provider
    id: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]*_[a-zA-Z0-9_]*$")
    cls: str  # Name of the LangChain class for the constructor
    model: str  # Name of the model for the constructor

    @computed_field
    def key(self) -> str:
        return API_KEYS[self.cls]

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        parts = v.split("_")
        if len(parts) != 3:
            raise ValueError(
                "id must have exactly 3 parts separated by underscores: model_version_provider"
            )
        return v


def read_llm_list_file() -> list[LlmInfo]:
    yml_file = Path(get_config_str("llm", "list"))
    assert yml_file.exists(), f"cannot find {yml_file}"
    with open(yml_file, "r") as f:
        data = yaml.safe_load(f)

    llms = []
    for provider in data["llm"]:
        cls = provider["cls"]
        for model in provider["models"]:
            model["cls"] = cls
            llms.append(LlmInfo(**model))
    return llms


class LlmFactory(BaseModel):
    """
    Factory for creating and configuring LLM instances.

    Handles the creation of LangChain BaseLanguageModel instances with appropriate
    configuration based on the model type and provider.

    Attributes:
        llm_id: Unique model identifier (if None, uses default from config)
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        max_tokens: Maximum tokens in model response
        json_mode: Whether to force JSON output format (where supported)
        cache: Optional LlmCache instance for response caching
        streaming: Whether to enable streaming responses (where supported)
    """

    llm_id: Annotated[str | None, Field(validate_default=True)] = None
    temperature: float = 0
    max_tokens: int = MAX_TOKENS
    json_mode: bool = False
    cache: LlmCache | None = None
    streaming: bool = False

    @computed_field
    @cached_property
    def info(self) -> LlmInfo:
        """Return LLM_INFO information on LLM"""
        assert self.llm_id
        return LlmFactory.known_items_dict().get(self.llm_id)  # type: ignore

    @field_validator("llm_id", mode="before")
    def check_known(cls, llm_id: str | None) -> str:
        if llm_id is None:
            llm_id = get_config_str("llm", "default_model")
        if llm_id not in LlmFactory.known_items():
            raise ValueError(
                f"Unknown LLM: {llm_id}; Should be in {LlmFactory.known_items()}"
            )
        return llm_id

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[LlmInfo]:
        return read_llm_list_file()

    @staticmethod
    def known_items_dict() -> dict[str, LlmInfo]:
        """Return known LLM in the registry whose API key environment variable is known"""
        return {
            item.id: item
            for item in LlmFactory.known_list()
            if item.key in os.environ or item.key == ""
        }

    @staticmethod
    def known_items() -> list[str]:
        """Return id of known LLM in the registry whose API key environment variable is known"""

        return list(LlmFactory.known_items_dict().keys())

    def get_id(self):
        assert self.llm_id
        return self.llm_id

    def short_name(self):
        return self.info.id.rsplit("_", maxsplit=1)[0]

    def get(self) -> BaseLanguageModel:
        """
        Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the known_llm_list() table, otherwise
        we create the LLM from a LangChain LLM class
        """
        if self.info.key not in os.environ and self.info.key != "":
            raise ValueError(f"No known API key for : {self.llm_id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> BaseLanguageModel:
        """Model factory, according to the model class"""
        if self.info.cls == "ChatOpenAI":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=42,
            )
            if self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
        elif self.info.cls == "ChatGroq":
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=None,
                streaming=self.streaming,
            )
            if self.streaming:
                llm.streaming = True
            if self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
        elif self.info.cls == "ChatDeepInfra":
            from langchain_community.chat_models.deepinfra import ChatDeepInfra

            llm = ChatDeepInfra(
                name=self.info.model,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )
            if True:  # self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
        elif self.info.cls == "ChatEdenAI":
            from langchain_community.chat_models.edenai import ChatEdenAI

            provider, _, model = self.info.model.partition("/")

            llm = ChatEdenAI(
                provider=provider,
                model=model,
                max_tokens=self.max_tokens,
                # edenai_api_url="https://staging-api.edenai.run/v2",
                edenai_api_key=None,  # set in env. variable
                temperature=self.temperature,
            )

        elif self.info.cls == "ChatVertexAI":
            from langchain_google_vertexai import ChatVertexAI  # type: ignore  # noqa: I001

            llm = ChatVertexAI(
                model=self.info.model,
                project="prj-p-eden",  # TODO : set it in config
                convert_system_message_to_human=True,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )  # type: ignore
            assert not self.json_mode, "json_mode not supported or coded"
        elif self.info.cls == "ChatOllama":
            from langchain_ollama import ChatOllama  # type: ignore

            format = "json" if self.json_mode else ""
            llm = ChatOllama(
                model=self.info.model, format=format, temperature=self.temperature
            )

            # llm = llama3_formatter | llm
        elif self.info.cls == "AzureChatOpenAI":
            from langchain_openai import AzureChatOpenAI

            name, _, api_version = self.info.model.partition("/")
            llm = AzureChatOpenAI(
                name=name,
                azure_deployment=name,
                model=name,  # Not sure it's needed
                api_version=api_version,
                temperature=self.temperature,
            )
            if self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
        elif self.info.cls == "ChatTogether":
            from langchain_together import ChatTogether  # type: ignore

            llm = ChatTogether(
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.info.cls == "ChatOpenrouter":
            from langchain_openai import ChatOpenAI

            OPENROUTER_BASE = "https://openrouter.ai"
            OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
            llm = ChatOpenAI(
                base_url=OPENROUTER_API_BASE,
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=42,
                api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore
                # headers={"HTTP-Referer": ...},
            )
            if self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )

        else:
            if self.info.cls in LlmFactory.known_items():
                raise ValueError(f"No API key found for LLM: {self.info.cls}")
            else:
                raise ValueError(f"unsupported LLM class {self.info.cls}")

        set_cache(self.cache)
        return llm

    def get_configurable(self, with_fallback=False) -> Runnable:
        # Make the LLM configurable at run time
        # see https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1

        default_llm_id = self.llm_id
        if default_llm_id is None:
            default_llm_id = get_config_str("llm", "default_model")

        # The field alternatives is created from our list of LLM
        alternatives = {}
        for llm_id in LlmFactory.known_items():
            if llm_id != default_llm_id:
                try:
                    llm_obj = LlmFactory(llm_id=llm_id).get()
                    alternatives |= {llm_id: llm_obj}
                except Exception as ex:
                    logger.info(f"Cannot load {llm_id}: {ex}")

        selected_llm = (
            LlmFactory(llm_id=self.llm_id)
            .get()
            .configurable_alternatives(  # select default LLM
                ConfigurableField(id="llm"),
                default_key=default_llm_id,
                prefix_keys=False,
                **alternatives,
            )
        )
        if with_fallback:
            # Not well tested !!!
            selected_llm = selected_llm.with_fallbacks(
                [LlmFactory(llm_id="llama3_70_groq").get()]
            )
        return selected_llm


def get_llm(
    llm_id: str | None = None,
    temperature: float = 0,
    max_tokens: int = MAX_TOKENS,
    json_mode: bool = False,
    cache: LlmCache | None = None,
    configurable: bool = False,
    with_fallback=False,
    streaming: bool = False,
) -> BaseLanguageModel:
    """
    Create a configured LangChain BaseLanguageModel instance.

    Args:
        llm_id: Unique model identifier (if None, uses default from config)
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        max_tokens: Maximum tokens in model response
        json_mode: Force JSON output format (where supported)
        cache: Optional LlmCache instance for response caching
        configurable: Make model configurable at runtime
        with_fallback: Add fallback model for reliability
        streaming: Enable streaming responses (where supported)

    Returns:
        BaseLanguageModel: Configured language model instance

    Raises:
        ValueError: If llm_id is invalid or API key is missing
    """
    factory = LlmFactory(
        llm_id=llm_id,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        cache=cache,
        streaming=streaming,
    )
    info = f"get LLM:'{factory.llm_id}'"
    info += " -configurable" if configurable else ""
    info += " -streaming" if streaming else ""
    info += " -json_mode" if json_mode else ""
    info += f" -cache: {cache}" if cache else ""

    logger.info(info)
    if configurable:
        return cast(
            BaseLanguageModel, factory.get_configurable(with_fallback=with_fallback)
        )
    else:
        return factory.get()


def get_selected_llm(args) -> BaseLanguageModel:
    return get_llm()


def get_llm_info(llm_id: str) -> LlmInfo:
    """
    Return information on given LLM
    """
    factory = LlmFactory(llm_id=llm_id)
    r = factory.known_items_dict().get(llm_id)
    if r is None:
        raise ValueError(f"Unknown llm_id: {llm_id} ")
    else:
        return r

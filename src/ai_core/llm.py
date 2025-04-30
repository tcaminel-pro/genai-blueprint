"""Language Model (LLM) factory and configuration management.

This module implements a factory pattern for creating and managing Language Learning Models
from various providers. It handles model configuration, runtime switching, and integration
with caching and streaming features.

Key Features:
- Support for multiple providers (OpenAI, DeepInfra, Groq, etc.)
- Runtime model switching and fallback mechanisms
- Structured JSON output support
- Caching and streaming capabilities
- Configuration through YAML files
- API key management via environment variables

Models are identified by unique IDs following the pattern: model_version_provider
Example: gpt_35_openai for GPT-3.5 from OpenAI

Configuration is stored in models_providers.yaml and supports:
- Default model selection
- API key management
- Cache configuration
- Streaming options

Example:
    >>> # Get default LLM
    >>> llm = get_llm()

    >>> # Get specific model with JSON output
    >>> llm = get_llm(llm_id="gpt_35_openai", json_mode=True)

    >>> # Get configurable LLM with fallback
    >>> llm = get_configurable_llm(with_fallback=True)
"""


# TODO
#  implement from langchain_core.rate_limiters import InMemoryRateLimiter

import functools
import importlib.util
import os
from functools import cached_property, lru_cache
from typing import Annotated, Any, cast

import yaml
from devtools import debug
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _SUPPORTED_PROVIDERS
from langchain.globals import get_llm_cache
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableField, RunnableConfig, RunnableLambda
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator
from smolagents import AzureOpenAIServerModel, LiteLLMModel
from smolagents.models import ApiModel

from src.ai_core.cache import LlmCache
from src.utils.config_mngr import global_config

# model = init_chat_model("llama3-8b-8192", model_provider="XX")
# Supported model providers are: bedrock_converse, openai, huggingface, google_anthropic_vertex, fireworks, together, mistralai, bedrock, deepseek, groq, google_vertexai, ollama, cohere, anthropic, google_genai, azure_openai


load_dotenv(verbose=True, override=True)


SEED = 42  # Arbitrary value....
DEFAULT_MAX_RETRIES = 2

OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
# cSpell: disable

# List of implemented LLM providers, with the Python class to be loaded, and the name of the API key environment variable
PROVIDER_INFO = {
    "fake": ("langchain_core", ""),
    "openai": ("langchain_openai", "OPENAI_API_KEY"),
    "deepinfra": ("langchain_community.chat_models.deepinfra", "DEEPINFRA_API_TOKEN"),
    "groq": ("langchain_groq", "GROQ_API_KEY"),
    "ollama": ("langchain_ollama", ""),
    "edenai": ("langchain_community.chat_models.edenai", "EDENAI_API_KEY"),
    "azure_openai": ("langchain_openai", "AZURE_OPENAI_API_KEY"),
    "together": ("langchain_together", "TOGETHER_API_KEY"),
    "deepseek": ("langchain_deepseek", "DEEPSEEK_API_KEY"),
    "openrouter": ("langchain_openai", "OPENROUTER_API_KEY"),
    "huggingface": ("langchain_huggingface", "HUGGINGFACEHUB_API_TOKEN"),
    # NOT TESTED:
    "bedrock": ("langchain_aws", "AWS_ACCESS_KEY_ID"),
    "anthropic": ("langchain_anthropic", "ANTHROPIC_API_KEY"),
    "google_vertexai": ("langchain_google_vertexai", "GOOGLE_API_KEY"),
}


class LlmInfo(BaseModel):
    """Description of an LLM model and its configuration.

    Attributes:
        id: Unique identifier in format model_version_provider (e.g. gpt_35_openai)
        provider: name of the provider
        model: Model identifier used by the provider
        key: API key environment variable name (computed from cls)
    """

    # an ID for the LLM; should follow Python variables constraints, and have 3 parts: model_version_provider
    id: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]*_[a-zA-Z0-9_]*$")
    provider: str
    model: str  # Name of the model for the constructor

    @computed_field
    def key(self) -> str:
        # return API key name
        return PROVIDER_INFO[self.provider][1]

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        parts = v.split("_")
        if len(parts) != 3:
            raise ValueError("id must have exactly 3 parts separated by underscores: model_version_provider")
        return v


def _read_llm_list_file() -> list[LlmInfo]:
    """Read the YAML file with list of LLM providers and info."""
    # The name of the file is in the configuration file
    yml_file = global_config().get_file_path("llm.list")
    with open(yml_file) as f:
        data = yaml.safe_load(f)

    llms = []
    for provider in data["llm"]:
        name = provider["provider"]
        for model in provider["models"]:
            model["provider"] = name
            llms.append(LlmInfo(**model))
    return llms


class LlmFactory(BaseModel):
    """Factory for creating and configuring LLM instances.

    Handles the creation of LangChain BaseLanguageModel instances with appropriate
    configuration based on the model type and provider.

    Attributes:
        llm_id: Unique model identifier (if None, uses default from config)
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        cache: cache method ("sqlite", "memory", no"_cache, ..) or "default", or None if no change (global setting)
        llm_params : other llm parameters (temperature, max_token, ....)
    """

    llm_id: Annotated[str | None, Field(validate_default=True)] = None
    json_mode: bool = False
    streaming: bool = False
    cache: str | None = None
    llm_params: dict = {}

    @property
    def provider(self) -> str:
        return self.info.id.rsplit("_")[2]

    @computed_field
    @cached_property
    def info(self) -> LlmInfo:
        """Return LLM_INFO information on LLM."""
        assert self.llm_id
        return LlmFactory.known_items_dict().get(self.llm_id)  # type: ignore

    @field_validator("llm_id", mode="before")
    def check_known(cls, llm_id: str | None) -> str:
        if llm_id is None:
            llm_id = global_config().get_str("llm.default_model")
        if llm_id not in LlmFactory.known_items():
            # TODO : have a more detailed error message
            raise ValueError(
                f"Unknown LLM: {llm_id}; Check API key and module imports. Should be in {LlmFactory.known_items()}"
            )
        return llm_id

    @field_validator("cache")
    def check_known_cache(cls, cache: str | None) -> None:
        if cache and cache not in LlmCache.values():
            raise ValueError(f"Unknown cache method: '{cache} '; Should be in {LlmCache.values()}")

    @lru_cache(maxsize=1)
    @staticmethod
    def known_list() -> list[LlmInfo]:
        return _read_llm_list_file()

    @staticmethod
    def known_items_dict(explain: bool = False) -> dict[str, LlmInfo]:
        """Return known LLM in the registry whose API key environment variable is known and module can be imported.

        If 'explain', add information on debug.trace
        """
        known_items = {}
        for item in LlmFactory.known_list():
            module_name, api_key = PROVIDER_INFO.get(item.provider, (None, None))
            assert module_name, f"No PROVIDER_INFO for LLM provider {item.provider}"
            if api_key in os.environ or api_key == "":
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    known_items[item.id] = item
                elif explain:
                    logger.debug(f"Module {module_name} for {item.provider} could not be imported.")
            elif explain:
                logger.debug(f"No API key {api_key} for {item.provider}")
        return known_items

    @staticmethod
    def known_items() -> list[str]:
        """Return id of known LLM in the registry whose API key environment variable is known and Python module installed."""
        return sorted(LlmFactory.known_items_dict().keys())

    @staticmethod
    def find_llm_id_from_type(llm_type: str) -> str:
        llm_id = global_config().get_str(f"llm.{llm_type}", default="default")
        if llm_id == "default":
            raise ValueError(f"Cannot find LLM of type type : '{llm_type}' (no key found in config file)")
        if llm_id not in LlmFactory.known_items():
            raise ValueError(f"Cannot find LLM '{llm_id}' of type : '{llm_type}'")
        return llm_id

    def get_id(self) -> str:
        """Return the id of the LLM."""
        assert self.llm_id
        return self.llm_id

    def short_name(self) -> str:
        """Return the name and version of the LLMn without the provider."""
        return self.info.id.rsplit("_", maxsplit=1)[0]

    def get_litellm_model_name(self) -> str:
        """Return the LiteLLM id string from our llm_id  (best effort).

        Not all cases covered.
        """
        if self.provider in ["openai"]:
            result = f"{self.info.model}"
        else:
            result = f"{self.provider}/{self.info.model}"

        try:
            get_llm_provider(result)
            # Note: LiteLLM mentions a 'get_valid_models' call, but not seems present

        except Exception as ex:
            raise ValueError(f"Incorrect or unknown LiteLLM provider for: '{result}'") from ex

        return result

    def get_smolagent_model(self) -> ApiModel:
        if self.provider in ["azure"]:
            name, _, api_version = self.info.model.partition("/")
            model = AzureOpenAIServerModel(
                model_id=name,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY=your_azure_openai_key_here"),
                api_version=api_version,
            )
        else:
            model = LiteLLMModel(model_id=self.get_litellm_model_name(), **self.llm_params)
        debug(model)
        return model

    def get(self) -> BaseChatModel:
        """Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the known_llm_list() table, otherwise
        we create the LLM from a LangChain LLM class.

        Example:
            ```python
            # Get default LLM
            llm = LlmFactory().get()

            # Get specific model with JSON output
            llm = LlmFactory(llm_id="gpt_35_openai", json_mode=True).get()

            # Generate a joke
            joke = llm.invoke("Tell me a joke about AI")
            ```
        """
        if self.info.key not in os.environ and self.info.key != "":
            raise ValueError(f"No known API key for : {self.llm_id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> BaseChatModel:
        """Model factory, according to the model class."""
        if self.cache:
            cache = LlmCache.from_value(self.cache)
        else:
            cache = get_llm_cache()

        common_params = {
            "temperature": 0.0,
            "cache": cache,
            "seed": SEED,
            "max_retries": DEFAULT_MAX_RETRIES,
            "streaming": self.streaming,
        }

        llm_params = common_params | self.llm_params
        if self.json_mode:
            llm_params |= {"response_format": {"type": "json_object"}}

        langchain_factory_supported_profider = set(_SUPPORTED_PROVIDERS)
        langchain_factory_supported_profider -= {"huggingface", "google_vertexai", "azure_openai"}

        if self.info.provider in langchain_factory_supported_profider:
            # Some parameters are handled differently between provider. Here some workaround:
            if self.info.provider in ["groq"]:
                seed = llm_params.pop("seed")
                llm_params |= {"model_kwargs": {"seed": seed}}
            llm = init_chat_model(model=self.info.model, model_provider=self.info.provider, **llm_params)

        elif self.info.provider == "deepinfra":
            from langchain_community.chat_models.deepinfra import ChatDeepInfra

            llm = ChatDeepInfra(
                name=self.info.model,
                **llm_params,
            )
        elif self.info.provider == "edenai":
            from langchain_community.chat_models.edenai import ChatEdenAI

            provider, _, model = self.info.model.partition("/")
            _ = llm_params.pop("seed")
            _ = llm_params.pop("json_object", None)
            _ = llm_params.pop("max_retries")

            llm = ChatEdenAI(
                provider=provider,
                model=model,
                edenai_api_key=None,  # set in env. variable
                **llm_params,
            )

        elif self.info.provider == "google_vertexai":
            from langchain_google_vertexai import ChatVertexAI  # type: ignore  # noqa: I001

            llm = ChatVertexAI(
                model=self.info.model,
                project="prj-p-eden",  # TODO : set it in config
                convert_system_message_to_human=True,
                **llm_params,
            )  # type: ignore
            assert not self.json_mode, "json_mode not supported or coded"

        elif self.info.provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI

            name, _, api_version = self.info.model.partition("/")
            llm = AzureChatOpenAI(
                name=name,
                azure_deployment=name,
                model=name,  # Not sure it's needed
                api_version=api_version,
                **llm_params,
            )
            if self.json_mode:
                llm = cast(BaseLanguageModel, llm.bind(response_format={"type": "json_object"}))
        elif self.info.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            # See https://openrouter.ai/docs/parameters

            # _ = llm_params.pop("response_format", None) or {}
            # Not sure.  See https://openrouter.ai/docs/structured-outputs
            llm = ChatOpenAI(
                base_url=OPENROUTER_API_BASE,
                model=self.info.model,
                api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore
                **llm_params,
            )
        elif self.info.provider == "huggingface":
            # NOT WELL TESTED
            # Also consider : https://huggingface.co/blog/inference-providers
            # see https://huggingface.co/blog/langchain
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            llm = HuggingFaceEndpoint(
                repo_id=self.info.model,
                task="text-generation",
                do_sample=False,
            )  # type: ignore
            return ChatHuggingFace(llm=llm)
        elif self.info.provider == "fake":
            from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel

            if self.info.model == "parrot":
                llm = ParrotFakeChatModel()
            else:
                raise ValueError(f"unsupported fake model {self.info.model}")

        else:
            if self.info.provider in LlmFactory.known_items():
                raise ValueError(f"No API key found for LLM: {self.info.provider}")
            else:
                raise ValueError(f"unsupported LLM class {self.info.provider}")

        return llm  # type: ignore

    def get_configurable(self, with_fallback: bool = False) -> BaseChatModel:
        # Make the LLM configurable at run time
        # see https://python.langchain.com/docs/how_to/configure/#configurable-alternatives

        default_llm_id = self.llm_id
        if default_llm_id is None:
            default_llm_id = global_config().get_str("llm.default_model")

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
                ConfigurableField(id="llm_id"),
                default_key=default_llm_id,
                prefix_keys=False,
                **alternatives,
            )
        )
        if with_fallback:
            # Not well tested !!!
            selected_llm = selected_llm.with_fallbacks([LlmFactory(llm_id="llama3_70_groq").get()])
        return selected_llm  # type: ignore


def get_llm(
    llm_id: str | None = None,
    llm_type: str | None = None,
    json_mode: bool = False,
    streaming: bool = False,
    cache: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create a configured LangChain BaseLanguageModel instance.

    Args:
        llm_id: Unique model identifier (if None, uses default from config)
        llm_type: Type of model to use (fast_model, smart_model, etc.)
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        cache: cache method ("sqlite", "memory", no"_cache, ..) or "default", or None if no change (global setting)
        **kwargs: other llm parameters (temperature, max_token, ....)

    Returns:
        BaseLanguageModel: Configured language model instance

    Examples:
        ```python
        # Get default LLM
        llm = get_llm()

        # Get specific model with streaming
        llm = get_llm(llm_id="gpt_35_openai", streaming=True)

        # Get model with custom temperature
        llm = get_llm(llm_id="llama3_70_groq", temperature=0.7)

        # Use in a chain
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        chain = prompt | get_llm(llm_id="gpt_4o_openai")
        result = chain.invoke({"topic": "AI"})
        ```
    """
    if llm_type and llm_id:
        logger.warning(
            "llm_type and llm_id both  defined whereas they are normally exclusive.  llm_id has the preference"
        )
    elif llm_type:
        llm_id = LlmFactory.find_llm_id_from_type(llm_type)

    factory = LlmFactory(
        llm_id=llm_id,
        json_mode=json_mode,
        streaming=streaming,
        cache=cache,
        llm_params=kwargs,
    )
    info = f"get LLM:'{factory.llm_id}'"
    info += " -streaming" if streaming else ""
    info += " -json_mode" if json_mode else ""
    info += f" -cache: {cache}" if cache else ""
    info += f" -extra: {kwargs}" if kwargs else ""
    logger.info(info)
    return factory.get()


def get_configurable_llm(
    json_mode: bool = False, with_fallback: bool = False, streaming: bool = False, cache: str | None = None, **kwargs
) -> BaseChatModel:
    """Create a configurable LangChain BaseLanguageModel instance.

    Args:
        json_mode: Whether to force JSON output format (where supported)
        streaming: Whether to enable streaming responses (where supported)
        cache: cache method ("sqlite", "memory", no"_cache, ..) or "default", or None if no change (global setting)
        **kwargs: other llm parameters (temperature, max_token, ....)

    Returns:
        BaseLanguageModel: Configured language model instance

    Example :
    .. code-block:: python
        from src.ai_core.prompts import def_prompt
        from src.ai_core.llm import get_configurable_llm, llm_config

        chain = def_prompt("tell me a joke") | get_configurable_llm()
        r = chain.with_config(llm_config("claude_haiku35_openrouter")).invoke({})
        # or:
        r = chain.invoke({}, config=llm_config("gpt_35_openai"))

    """
    factory = LlmFactory(llm_id=None, json_mode=json_mode, streaming=streaming, cache=cache, llm_params=kwargs)

    info = f"get configurable LLM. Default is:'{factory.llm_id}'"
    info += " -streaming" if streaming else ""
    info += " -json_mode" if json_mode else ""
    info += f" -cache: {cache}" if cache else ""
    info += f" -extra: {kwargs}" if kwargs else ""

    logger.info(info)
    return factory.get_configurable(with_fallback=with_fallback)


# workaround  to cache the function while keeping its signature.  See :
# https://github.com/python/typeshed/issues/11280#issuecomment-1987620682
# todo : consider https://github.com/umarbutler/persist-cache
# TODO: Does not work with Python 3.12
get_configurable_llm = functools.wraps(get_configurable_llm)(functools.cache(get_configurable_llm))


def get_llm_info(llm_id: str) -> LlmInfo:
    """Return information on given LLM."""
    factory = LlmFactory(llm_id=llm_id)
    r = factory.known_items_dict().get(llm_id)
    if r is None:
        raise ValueError(f"Unknown llm_id: '{llm_id}' ")
    else:
        return r


def llm_config(llm_id: str) -> RunnableConfig:
    """Return a 'RunnableConfig' to configure an LLM at run-time. Check LLM is known.

    Examples :
    ```
        r = chain.with_config(llm_config("claude_haiku35_openrouter")).invoke({})
        # or:
        r = graph.invoke({}, config=llm_config("gpt_35_openai") | {"recursion_limit": 6}) )
    ```
    """
    if llm_id not in LlmFactory.known_items():
        raise ValueError(
            f"Unknown LLM: {llm_id}; Check API key and module imports. Should be in {LlmFactory.known_items()}"
        )
    return configurable({"llm_id": llm_id})


def configurable(conf: dict) -> RunnableConfig:
    """Return a dict with key 'configurable', to be used in 'with_config'.

    Example:
    ```
        llm.with_config(configurable({"my_conf": "my_conf_value"})  )

    """
    return {"configurable": conf}


def get_print_chain(string: str = "") -> RunnableLambda:
    """Return a chain that print the passed input and the config. Useful for debugging.

    Example:
    ```
        from src.ai_core.llm import configurable, get_print_chain

        add_1 = get_print_chain("before") | RunnableLambda(lambda x: x + 1) | get_print_chain("after")
        chain = add_1.with_config(configurable({"my_conf": "my_conf_value"}))
        print(chain.invoke(1))
    ```
    """

    def fn(input: Any, config: RunnableConfig) -> Any:
        print(string, input, config)
        return input

    return RunnableLambda(fn)

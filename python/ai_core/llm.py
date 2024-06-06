"""
LLM model factory.

Facilitate the creation of Llm objects, that are :
- Configurable (we can change the LLM)
 -With fallback

The LLM can be given, or taken from the configuration
"""

import os
from functools import cache, cached_property
from typing import cast

from langchain.globals import set_llm_cache
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_core.runnables import ConfigurableField, Runnable
from loguru import logger
from lunary import LunaryCallbackHandler
from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.ai_core.agents_builder import AgentBuilder, get_agent_builder
from python.config import get_config_str

MAX_TOKENS = 2048


class LLM_INFO(BaseModel):
    id: str  # an ID for the LLM; should follow Python variables constraints
    cls: str  # Name of the LangChain class for the constructor
    model: str  # Name of the model for the constructor
    key: str  # API key.  "" if it not needed.
    agent_builder_type: str = "tool_calling"

    @computed_field
    @property
    def agent_builder(self) -> AgentBuilder:
        return get_agent_builder(self.agent_builder_type)

    @field_validator("agent_builder")
    def check_known(cls, type: str) -> str:
        _ = get_agent_builder(type)
        return type

    def __hash__(self):
        return hash(self.id)

    # DODO: add validator for LLM id name.


KNOWN_LLM_LIST = [
    # LLM id should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{self.model_name}_{version}_{inference provider or library}"
    # model_name is provider specific.  It can contains several fields decoded in the factory.
    # LiteLlm supported models are listed here: https://litellm.vercel.app/docs/providers
    #
    # ####  OpenAI Models  ####
    LLM_INFO(
        id="gpt_35_openai",
        cls="ChatOpenAI",
        model="gpt-3.5-turbo-0125",
        key="OPENAI_API_KEY",
    ),
    #
    ####  ChatDeepInfra ### https://deepinfra.com/models/text-generation
    LLM_INFO(
        id="llama3_70_deepinfra",
        cls="ChatDeepInfra",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        key="DEEPINFRA_API_TOKEN",
    ),
    LLM_INFO(
        id="llama3_8_deepinfra",
        cls="ChatDeepInfra",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        key="DEEPINFRA_API_TOKEN",
    ),
    LLM_INFO(
        id="mixtral_7x8_deepinfra",
        cls="ChatDeepInfra",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        key="DEEPINFRA_API_TOKEN",
    ),
    #
    ####  ChatLiteLLM Models
    LLM_INFO(
        id="llama3_70_deepinfra_lite",
        cls="ChatLiteLLM",
        model="deepinfra/meta-llama/Llama-3-70b-chat-hf",
        key="DEEPINFRA_API_TOKEN",
    ),
    #
    #####  GROQ  Models  #####
    LLM_INFO(
        id="llama3_70_groq",
        cls="ChatGroq",
        model="lLama3-70b-8192",
        key="GROQ_API_KEY",
    ),
    LLM_INFO(
        id="llama3_8_groq",
        cls="ChatGroq",
        model="lLama3-8b-8192",
        key="GROQ_API_KEY",
    ),
    LLM_INFO(
        id="mixtral_7x8_groq",
        cls="ChatGroq",
        model="Mixtral-8x7b-32768",
        key="GROQ_API_KEY",
        agent_builder_type="tool_calling",  # DOES NOT WORK # TODO : Check with new updates
    ),
    #
    #  Google Models
    LLM_INFO(
        id="gemini_pro_google",
        cls="ChatVertexAI",
        model="gemini-pro",
        key="GOOGLE_API_KEY",
    ),
    #
    #  Ollama Models
    LLM_INFO(
        id="llama3_8_local",
        cls="ChatOllama",
        model="llama3:instruct",
        key="",
    ),
    #
    ###  EdenAI Endpoint - see https://app.edenai.run/bricks/text/chat
    LLM_INFO(
        id="gpt_4o_edenai",
        cls="ChatEdenAI",
        model="openai/gpt-4o",
        key="EDENAI_API_KEY",
    ),
    LLM_INFO(
        id="gpt_4_edenai",
        cls="ChatEdenAI",
        model="openai/gpt-4",
        key="EDENAI_API_KEY",
    ),
    LLM_INFO(
        id="gpt_35_edenai",
        cls="ChatEdenAI",
        model="openai/gpt-3.5-turbo-0125",
        key="EDENAI_API_KEY",
    ),
    LLM_INFO(
        id="mistral_large_edenai",
        cls="ChatEdenAI",
        model="mistral/large-latest",
        key="EDENAI_API_KEY",
    ),
    LLM_INFO(
        id="gpt_4o_azure",
        cls="AzureChatOpenAI",
        model="gpt-4o/024-05-13",
        key="AZURE_OPENAI_API_KEY",
    ),
]


class LlmFactory(BaseModel):
    llm_id: Annotated[str | None, Field(validate_default=True)] = None
    temperature: float = 0
    max_tokens: int = MAX_TOKENS
    json_mode: bool = False
    cache: bool | None = None

    @computed_field
    @cached_property
    def info(self) -> LLM_INFO:
        assert self.llm_id
        return LlmFactory.known_items_dict().get(self.llm_id)  # type: ignore

    @field_validator("llm_id", mode="before")
    def check_known(cls, llm_id: str) -> str:
        if llm_id is None:
            llm_id = get_config_str("llm", "default_model")
        if llm_id not in LlmFactory.known_items():
            raise ValueError(f"Unknown LLM: {llm_id}")
        return llm_id

    @staticmethod
    def known_items_dict() -> dict[str, LLM_INFO]:
        return {
            item.id: item
            for item in KNOWN_LLM_LIST
            if item.key in os.environ or item.key == ""
        }

    @staticmethod
    def known_items() -> list[str]:
        return list(LlmFactory.known_items_dict().keys())

    def get(self) -> BaseLanguageModel:
        """
        Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the KNOWN_LLM_LIST table, otherwise
        we create the LLM from a LangChain LLM class
        """
        if self.info.key not in os.environ and self.info.key != "":
            raise ValueError(f"No known API key for : {self.llm_id}")
        llm = self.model_factory()
        return llm

    def model_factory(self) -> BaseLanguageModel:
        if self.info.cls == "ChatOpenAI":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={"seed": 42},  # Not sure that works
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
            )
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
            assert not self.json_mode, "json_mode not supported or coded"
        elif self.info.cls == "ChatEdenAI":
            from langchain_community.chat_models.edenai import ChatEdenAI

            provider, _, model = self.info.model.partition("/")

            llm = ChatEdenAI(
                provider=provider,
                model=model,
                max_tokens=self.max_tokens,
                edenai_api_url="https://staging-api.edenai.run/v2",
                edenai_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYTczYjA4OGMtOWE5MS00ODdhLWEyNTQtODJmNDI5MDVlNjhmIiwidHlwZSI6ImFwaV90b2tlbiJ9.KGkM6Sqtd0pD-83twNtUdSFtrNJD_u9ZZX3tabcDloA",  # or use `EDENAI_API_KEY` env var
                temperature=self.temperature,
            )

        elif self.info.cls == "ChatVertexAI":
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(
                model=self.info.model,
                project="prj-p-eden",  # TODO : set it in config
                convert_system_message_to_human=True,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )  # type: ignore
            assert not self.json_mode, "json_mode not supported or coded"
        elif self.info.cls == "ChatLiteLLM":
            from langchain_community.chat_models.litellm import ChatLiteLLM

            llm = ChatLiteLLM(
                model=self.info.model,
                temperature=self.temperature,
            )  # type: ignore

        elif self.info.cls == "ChatOllama":
            from langchain_community.chat_models.ollama import ChatOllama

            format = "json" if self.json_mode else None
            llm = ChatOllama(
                model=self.info.model, format=format, temperature=self.temperature
            )

            # llm = llama3_formatter | llm
        elif self.info.cls == "AzureChatOpenAI":
            from langchain_openai import AzureChatOpenAI

            name, _, version = self.info.model.partition("/")

            llm = AzureChatOpenAI(
                name=name,
                api_version=version,
                model=self.info.model,
                temperature=self.temperature,
            )

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
    cache: bool | None = None,
    configurable: bool = True,
    with_fallback=False,
) -> BaseLanguageModel:
    """
    Create a BaseLanguageModel object according to a given llm_id.\n
    - If 'llm_id' is None, the LLM is selected from the configuration.
    - 'json_mode' is not supported or tested for all models
    - 'configurable' make the LLM configurable at run-time
    - 'with_fallback' add a fallback mechanism (not well tested)

    """
    factory = LlmFactory(
        llm_id=llm_id,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        cache=cache,
    )
    logger.info(f"get LLM : {factory.llm_id} - configurable: {configurable}")
    if configurable:
        return cast(
            BaseLanguageModel, factory.get_configurable(with_fallback=with_fallback)
        )
    else:
        return factory.get()


def get_selected_llm(args) -> BaseLanguageModel:
    return get_llm()


def get_llm_info(llm_id: str) -> LLM_INFO:
    """
    Return information on given LLM
    """
    factory = LlmFactory(llm_id=llm_id)
    r = factory.known_items_dict().get(llm_id)
    if r is None:
        raise ValueError(f"Unknown llm_id: {llm_id} ")
    else:
        return r


@cache
def set_cache(cache: str | None = None):
    """
    Define caching strategy.  If 'None', take the one defined in configuration
    """
    if not cache:
        cache = get_config_str("llm", "cache")
    elif cache == "memory":
        set_llm_cache(InMemoryCache())
    elif cache == "sqlite":
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    else:
        raise ValueError(
            "incorrect [llm]/cache config. Should be 'memory' or 'sqlite' "
        )
    # logger.info(f"cache: {cache}")


llm_monitor_handler = LunaryCallbackHandler()  # Not used yet

"""
LLM self.models factory and Runnable

"""

import os
from functools import cache, cached_property
from typing import Type, cast

from langchain.globals import set_llm_cache
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.llms.deepinfra import DeepInfra
from langchain_community.llms.edenai import EdenAI
from langchain_core.runnables import ConfigurableField, Runnable
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from lunary import LunaryCallbackHandler
from pydantic import BaseModel, Field, computed_field, field_validator
from typing_extensions import Annotated

from python.ai_core.agents_builder import AgentBuilder, get_agent_builder
from python.config import get_config

MAX_TOKENS = 2048


class LLM_INFO(BaseModel):
    id: str
    cls: Type[BaseLanguageModel]
    model: str
    key: str
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


KNOWN_LLM_LIST = [
    # LLM id should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{self.model name}_{version}_{inference provider or library}"
    # LiteLlm supported models are listed here: https://litellm.vercel.app/docs/providers
    LLM_INFO(
        id="gpt_35_openai",
        cls=ChatOpenAI,
        model="gpt-3.5-turbo-0125",
        key="OPENAI_API_KEY",
    ),
    LLM_INFO(
        id="llama3_70_deepinfra",
        cls=DeepInfra,
        model="meta-llama/Llama-3-70b-chat-hf",
        key="DEEPINFRA_API_TOKEN",
    ),
    LLM_INFO(
        id="llama3_70_deepinfra_lite",
        cls=ChatLiteLLM,
        model="deepinfra/meta-llama/Llama-3-70b-chat-hf",
        key="DEEPINFRA_API_TOKEN",
    ),
    LLM_INFO(
        id="llama3_70_groq",
        cls=ChatGroq,
        model="lLama3-70b-8192",
        key="GROQ_API_KEY",
    ),
    LLM_INFO(
        id="gpt_35_edenai",
        key="EDENAI_API_KEY",
        cls=EdenAI,
        model="openai/gpt-3.5-turbo-instruct",
    ),
    LLM_INFO(
        id="llama3_8_groq",
        cls=ChatGroq,
        model="lLama3-8b-8192",
        key="GROQ_API_KEY",
    ),
    LLM_INFO(
        id="mixtral_7x8_groq",
        cls=ChatGroq,
        model="Mixtral-8x7b-32768",
        key="GROQ_API_KEY",
        agent_builder_type="openai_function",  # DOES NOT WORK # TODO : Check with new updates
    ),
    LLM_INFO(
        id="gemini_pro_google",
        cls=ChatVertexAI,
        model="gemini-pro",
        key="GOOGLE_API_KEY",
    ),
    LLM_INFO(
        id="llama3_8_local",
        cls=ChatOllama,
        model="llama3:instruct",
        key="",
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
            llm_id = get_config("llm", "default_model")
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
        if self.info.cls == ChatOpenAI:
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
        elif self.info.cls == ChatGroq:
            llm = ChatGroq(
                model=self.info.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if self.json_mode:
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
        elif self.info.cls == DeepInfra:
            llm = DeepInfra(
                model_id=self.info.model,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )
            assert not self.json_mode, "json_mode not supported or coded"
        elif self.info.cls == EdenAI:
            provider, _, model = self.info.model.partition("/")
            llm = EdenAI(
                feature="text",
                provider=provider,
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            assert not self.json_mode, "json_mode not supported"
        elif self.info.cls == ChatVertexAI:
            llm = ChatVertexAI(
                model=self.info.model,
                project="prj-p-eden",  # TODO : set it in config
                convert_system_message_to_human=True,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )  # type: ignore
            assert not self.json_mode, "json_mode not supported or coded"
        elif self.info.cls == ChatLiteLLM:
            llm = ChatLiteLLM(
                model=self.info.model,
                temperature=self.temperature,
            )  # type: ignore

        elif self.info.cls == ChatOllama:
            format = "json" if self.json_mode else None
            llm = ChatOllama(
                model=self.info.model, format=format, temperature=self.temperature
            )

            # llm = llama3_formatter | llm
        else:
            raise ValueError(f"unsupported LLM class {self.info.cls}")

        set_cache(self.cache)
        return llm

    def get_configurable(self, with_fallback=False) -> Runnable:
        #
        # see https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1

        default_llm_id = self.llm_id
        if default_llm_id is None:
            default_llm_id = get_config("llm", "default_model")
        alternatives = {
            llm: LlmFactory(llm_id=llm).get()
            for llm in LlmFactory.known_items()
            if llm != default_llm_id
        }
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
            # Not tested
            selected_llm = selected_llm.with_fallbacks(
                [LlmFactory(llm_id="llama3_70_groq").get()]
            )
        return selected_llm


@cache
def get_llm(
    llm_id: str | None = None,
    temperature: float = 0,
    max_tokens: int = MAX_TOKENS,
    json_mode: bool = False,
    cache: bool | None = None,
    configurable: bool = True,
    with_fallback=False,
) -> Runnable:
    factory = LlmFactory(
        llm_id=llm_id,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        cache=cache,
    )
    if configurable:
        return factory.get_configurable(with_fallback=with_fallback)
    else:
        return factory.get()


def get_llm_info(llm_id: str) -> LLM_INFO:
    r = LlmFactory.known_items_dict().get(llm_id)
    if r is None:
        raise ValueError(f"Unknown llm_id: {llm_id} ")
    else:
        return r


@cache
def set_cache(cache: str | None = None):
    if not cache:
        cache = get_config("llm", "cache")
    elif cache == "memory":
        set_llm_cache(InMemoryCache())
    elif cache == "sqlite":
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    else:
        raise ValueError(
            "incorrect [llm]/cache config. Should be 'memory' or 'sqlite' "
        )


llm_monitor_handler = LunaryCallbackHandler()  # Not used yet

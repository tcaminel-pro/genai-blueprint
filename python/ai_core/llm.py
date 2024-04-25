"""
LLM self.models factory and Runnable

"""

from abc import ABC, abstractmethod
import os
from functools import cache
from typing import cast

from pydantic import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.runnables import Runnable, ConfigurableField

from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

from langchain_community.llms.deepinfra import DeepInfra
from langchain_community.llms.edenai import EdenAI
from langchain_groq import ChatGroq

from langchain_core.runnables import (
    RunnableLambda,
)

from langchain_community.chat_models.litellm import ChatLiteLLM
from lunary import LunaryCallbackHandler
from python.config import get_config


MAX_TOKENS = 2048


class LLM_INFO(BaseModel):
    id: str
    litellm: str | None = None

    def __hash__(self):
        return hash(self.id)


KNOWN_LLM_LIST = [
    # LLM id should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{self.model name}_{version}_{inference provider or library}"
    LLM_INFO(id="gpt_35_openai"),
    LLM_INFO(id="gpt_35_edenai"),
    LLM_INFO(
        id="llama2_70_deepinfra", litellm="deepinfra/meta-llama/Llama-2-70b-chat-hf"
    ),
    LLM_INFO(id="llama3_70_groq", litellm="groq/llama3-70b-8192"),
    LLM_INFO(id="mixtral_7x8_deepinfra"),
    LLM_INFO(id="mixtral_7x8_groq", litellm="groq/mixtral-8x7b-32768"),
]

KNOWN_LLM_TABLE = {llm.id: llm for llm in KNOWN_LLM_LIST}
KNOWN_LLM = KNOWN_LLM_TABLE.keys()


class LlmFactory(BaseModel):
    llm_id: str | None = None
    temperature: float = 0
    max_tokens: int = MAX_TOKENS
    json_mode: bool = False

    def get(self, llm_id: str | None = None) -> BaseLanguageModel:
        """
        Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the KNOWN_LLM_LIST table, otherwier
        we create the LLM from a LangChain LLM class
        """

        if llm_id is None:
            llm_id = self.llm_id
        if self.llm_id is None:
            llm_id = get_config("llm", "default_model")
        self.llm_id = llm_id
        assert self.llm_id

        llm_info = KNOWN_LLM_TABLE.get(self.llm_id)

        if not llm_info:
            raise ValueError(f"Unknown LLM : {llm_id}")
        if llm_info.litellm:
            llm = ChatLiteLLM(model=llm_info.litellm, client=None)
        else:
            llm = self.custom_model_factory()
        return llm

    def custom_model_factory(self) -> BaseLanguageModel:
        """
        Create an LLM model using LangChain provided modules.
        """

        if self.llm_id == "gpt_35_openai":
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-0125",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={"seed": 42},  # Not sure that works
            )

        elif self.llm_id == "llama2_70_deepinfra":
            llm = DeepInfra(
                model_id="meta-llama/Llama-2-70b-chat-hf",
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )

        elif self.llm_id == "mixtral_7x8_deepinfra":
            llm = DeepInfra(
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )
        elif self.llm_id == "mixtral_7x8_deepinfra_oai":
            key = os.getenv("DEEPINFRA_API_TOKEN")
            assert key, "No DEEPINFRA_API_TOKEN key found"
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-0125",
                base_url="meta-llama/Llama-2-70b-chat-hf",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=os.getenv("DEEPINFRA_API_TOKEN"),
                model_kwargs={"seed": 42},  # Not sure that works
            )

        elif self.llm_id == "llama2_70_groq":
            llm = ChatGroq(
                model="lLama2-70b-4096",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.llm_id == "llama3_70_groq":
            llm = ChatGroq(
                model="lLama3-70b-8192",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.llm_id == "mixtral_7x8_groq":
            llm = ChatGroq(
                model="Mixtral-8x7b-32768",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.llm_id == "gpt_35_edenai":
            llm = EdenAI(
                feature="text",
                provider="openai",
                model="gpt-3.5-turbo-instruct",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        else:
            raise ValueError(f"unsupported LLM: '{self.llm_id}'")

        if self.json_mode:  # NOT TESTED
            # see also https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html

            if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatGroq):
                llm = cast(
                    BaseLanguageModel, llm.bind(response_format={"type": "json_object"})
                )
            else:
                raise ValueError(f"json_mode  not supported for {type(llm)}")

        return llm

    def get_dynamic(self) -> BaseLanguageModel:
        #
        # see https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1

        default_llm_id = self.llm_id
        if default_llm_id is None:
            default_llm_id = get_config("llm", "default_model")
        alternatives = {
            llm: self.get(llm_id=llm) for llm in KNOWN_LLM if llm != default_llm_id
        }
        selected_llm = (
            self.get(default_llm_id)  # select default LLM
            .with_fallbacks(
                [self.get(llm_id="llama3_70_groq")]
            )  # To be changed, and set temperature etc
            .configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key=default_llm_id,
                prefix_keys=False,
                **alternatives,
            )
        )
        return selected_llm

    # def get_dynamic_formatted(self) -> Runnable:
    #     if self.llm_id is None:
    #         self.llm_id = get_config("llm", "default_model")
    #     if self.llm_id.startswith("llama3"):
    #         return self.get_dynamic() | RunnableLambda(
    #             lambda x: Llama3Format().to_chat_prompt(x)
    #         )
    #     else:
    #         return self.get_dynamic()


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

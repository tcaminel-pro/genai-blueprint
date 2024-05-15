"""
LLM self.models factory and Runnable

"""

import os
from functools import cache
from typing import cast

from langchain.globals import set_llm_cache
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_community.llms.deepinfra import DeepInfra
from langchain_community.llms.edenai import EdenAI
from langchain_core.runnables import ConfigurableField, Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from lunary import LunaryCallbackHandler
from pydantic import BaseModel

from python.config import get_config

MAX_TOKENS = 2048


AGENT_TYPES = [
    "tool-calling",
    "openai-tools",
    "openai-functions",
    "zero-shot-react-description",
]


class LLM_INFO(BaseModel):
    id: str
    litellm: str | None = None
    key: str

    def __hash__(self):
        return hash(self.id)


KNOWN_LLM_LIST = [
    # LLM id should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{self.model name}_{version}_{inference provider or library}"
    # LiteLlm supported models are listed here: https://litellm.vercel.app/docs/providers
    LLM_INFO(id="gpt_35_openai", key="OPENAI_API_KEY"),
    LLM_INFO(id="gpt_4o_openai", key="OPENAI_API_KEY"),
    LLM_INFO(id="gpt_35_openai_lite", litellm="gpt-3.5-turbo", key="OPENAI_API_KEY"),
    LLM_INFO(id="gpt_35_edenai", key="EDENAI_API_KEY"),
    LLM_INFO(
        id="llama2_70_deepinfra",
        litellm="deepinfra/meta-llama/Llama-2-70b-chat-hf",
        key="DEEPINFRA_API_TOKEN",
    ),
    LLM_INFO(
        id="llama3_70_groq", litellm=None, key="GROQ_API_KEY"
    ),  # "groq/llama3-70b-8192"
    LLM_INFO(
        id="llama3_8_groq", litellm=None, key="GROQ_API_KEY"
    ),  # "groq/llama3-8b-8192"
    LLM_INFO(id="mixtral_7x8_deepinfra", key="DEEPINFRA_API_TOKEN"),
    LLM_INFO(
        id="mixtral_7x8_groq", litellm="groq/mixtral-8x7b-32768", key="GROQ_API_KEY"
    ),
    LLM_INFO(id="gemini_pro_google", key="GOOGLE_API_KEY"),
]


class LlmFactory(BaseModel):
    llm_id: str | None = None
    temperature: float = 0
    max_tokens: int = MAX_TOKENS
    json_mode: bool = False
    cache: bool | None = None

    @staticmethod
    def known_llm_table() -> dict[str, LLM_INFO]:
        return {llm.id: llm for llm in KNOWN_LLM_LIST if llm.key in os.environ}

    @staticmethod
    def known_llm() -> list[str]:
        return list(LlmFactory.known_llm_table().keys())

    def get(self) -> BaseLanguageModel:
        """
        Create an LLM model.
        'model' is our internal name for the model and its provider. If None, take the default one.
        We select a LiteLLM wrapper if it's defined in the KNOWN_LLM_LIST table, otherwise
        we create the LLM from a LangChain LLM class
        """
        if self.llm_id is None:
            self.llm_id = get_config("llm", "default_model")
        assert self.llm_id

        llm_info = LlmFactory.known_llm_table().get(self.llm_id)
        assert llm_info

        if llm_info.key not in os.environ:
            raise ValueError(f"No known API key for : {self.llm_id}")

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
        elif self.llm_id == "gpt_4o_openai":
            llm = ChatOpenAI(
                model="gpt-4o",
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
        elif self.llm_id == "gemini_pro_google_genai":
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                convert_system_message_to_human=True,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )  # type: ignore
        elif self.llm_id == "gemini_pro_google":
            llm = ChatVertexAI(
                model="gemini-pro",
                project="prj-p-eden",
                convert_system_message_to_human=True,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )  # type: ignore

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

        set_cache(self.cache)
        return llm

    def get_configurable(self, with_fallback=True) -> Runnable:
        #
        # see https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1

        default_llm_id = self.llm_id
        if default_llm_id is None:
            default_llm_id = get_config("llm", "default_model")
        alternatives = {
            llm: self.get(llm_id=llm)
            for llm in LlmFactory.known_llm()
            if llm != default_llm_id
        }
        selected_llm = self.get(
            default_llm_id
        ).configurable_alternatives(  # select default LLM
            ConfigurableField(id="llm"),
            default_key=default_llm_id,
            prefix_keys=False,
            **alternatives,
        )
        if with_fallback:
            # Not tested
            selected_llm = selected_llm.with_fallbacks(
                [self.get(llm_id="llama3_70_groq")]
            )
        return selected_llm


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

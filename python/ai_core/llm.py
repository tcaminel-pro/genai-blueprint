"""
LLM self.models factory and Runnable

"""

import os
from functools import cache

from pydantic import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.runnables import Runnable, ConfigurableField

from langchain_experimental.chat_models.llm_wrapper import Llama2Chat, Mixtral
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

from langchain_community.llms.deepinfra import DeepInfra
from langchain_community.llms.edenai import EdenAI
from langchain_groq import ChatGroq

from langchain_core.runnables import (
    RunnablePassthrough,
    chain,
    RunnableParallel,
    RunnableLambda,
)

# from langchain_community.chat_models.litellm import ChatLiteLLM
# from langchain_community.chat_models.litellm import ChatLiteLLM
from lunary import LunaryCallbackHandler
from python.ai_utils.prompt_formatter import Llama3Format, OpenAIFormat
from python.config import get_config


MAX_TOKENS = 2048

KNOWN_LLM = {
    # Names should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{self.model name}_{version}_{inference provider or library}"
    "gpt_35_openai",
    "gpt_35_edenai",
    "llama2_70_deepinfra",
    "llama2_70_groq",
    "llama3_70_groq",
    "mixtral_7x8_deepinfra",
    "mixtral_7x8_groq",
}


class LlmFactory(BaseModel):
    model: str | None = None
    temperature: float = 0
    max_tokens: int = MAX_TOKENS
    json_mode: bool = False

    def get(self, model: str | None = None) -> BaseLanguageModel:
        """
        Create an LLM self.model.
        'model' is our internal name for the model and its provider. If None, take the default one
        """

        if model is None:
            model = self.model
        if self.model is None:
            model = get_config("llm", "default_model")
        self.model = model

        assert self.model

        if self.model == "gpt_35_openai":
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-0125",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={"seed": 42},  # Not sure that works
            )

        elif self.model == "llama2_70_deepinfra":
            llm = DeepInfra(
                model_id="meta-llama/Llama-2-70b-chat-hf",
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )

        elif self.model == "mixtral_7x8_deepinfra":
            llm = DeepInfra(
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "repetition_penalty": 1.3,
                    "stop": ["STOP_TOKEN"],
                },
            )
        elif self.model == "mixtral_7x8_deepinfra_oai":
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

        elif self.model == "llama2_70_groq":
            llm = ChatGroq(
                model="lLama2-70b-4096",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.model == "llama3_70_groq":
            llm = ChatGroq(
                model="lLama3-70b-8192",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.model == "mixtral_7x8_groq":
            llm = ChatGroq(
                model="Mixtral-8x7b-32768",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.model == "mixtral_7x8_groq_lite":
            llm = ChatLiteLLM(model="groq/mixtral-8x7b-32768", client=None)

        elif self.model == "gpt_35_edenai":
            llm = EdenAI(
                feature="text",
                provider="openai",
                model="gpt-3.5-turbo-instruct",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        else:
            raise ValueError(f"unsupported LLM: '{self.model}'")

        if self.json_mode:  # NOT TESTED
            # see also https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html

            if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatGroq):
                llm = llm.bind(response_format={"type": "json_object"})
            else:
                raise ValueError(f"json_mode  not supported for {type(llm)}")

        return llm

    def get_dynamic(self) -> Runnable:
        """
        Define a configurable LLM runnable

        https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1
        """
        default = get_config("llm", "default_model")
        alternatives = {llm: self.get(model=llm) for llm in KNOWN_LLM if llm != default}
        selected_llm = (
            self.get()  # select default LLM
            .with_fallbacks(
                [LlmFactory(model="gpt_35_openai").get()]
            )  # To be changed, and set temperature etc
            .configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key=default,
                prefix_keys=False,
                **alternatives,
            )
        )

        return selected_llm

    def get_dynamic_formatted(self) -> Runnable:
        if self.model is None:
            self.model = get_config("llm", "default_model")
        dev_debug(self.model)
        if self.model.startswith("llama3"):
            return self.get_dynamic() | RunnableLambda(
                lambda x: Llama3Format().to_chat_prompt(x)
            )
        else:
            return self.get_dynamic()


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

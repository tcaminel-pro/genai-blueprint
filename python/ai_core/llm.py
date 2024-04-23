"""
LLM models factory and Runnable

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

# from langchain_community.chat_models.litellm import ChatLiteLLM
from lunary import LunaryCallbackHandler
from python.config import get_config


MAX_TOKENS = 2048

KNOWN_LLM = {
    # Names should follow Python variables constraints - ie no '-', no space, etc
    # Use pattern "{model name}_{version}_{inference provider or library}"
    "gpt_35_openai",
    "gpt_35_edenai",
    "llama2_70_deepinfra",
    "llama2_70_groq",
    "llama3_70_groq",
    "mixtral_7x8_deepinfra",
    "mixtral_7x8_groq",
}


def llm_factory(
    model: str | None = None, temperature=0, max_tokens=MAX_TOKENS, json_mode=False
) -> BaseLanguageModel:
    """
    Create an LLM model.
    'model' is our internal name for the model and its provider. If None, take the default one
    """
    if model is None:
        model = get_config("llm", "default_model")

    if model == "gpt_35_openai":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"seed": 42},  # Not sure that works
        )

    elif model == "llama2_70_deepinfra":
        llm = DeepInfra(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )

    elif model == "mixtral_7x8_deepinfra":
        llm = DeepInfra(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )
    elif model == "mixtral_7x8_deepinfra_oai":
        key = os.getenv("DEEPINFRA_API_TOKEN")
        assert key, "No DEEPINFRA_API_TOKEN key found"
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            base_url="meta-llama/Llama-2-70b-chat-hf",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("DEEPINFRA_API_TOKEN"),
            model_kwargs={"seed": 42},  # Not sure that works
        )

    elif model == "llama2_70_groq":
        llm = ChatGroq(
            model="lLama2-70b-4096",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model == "llama3_70_groq":
        llm = ChatGroq(
            model="lLama3-70b-8192",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model == "mixtral_7x8_groq":
        llm = ChatGroq(
            model="Mixtral-8x7b-32768",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model == "mixtral_7x8_groq_lite":
        llm = ChatLiteLLM(model="groq/mixtral-8x7b-32768", client=None)

    elif model == "gpt_35_edenai":
        llm = EdenAI(
            feature="text",
            provider="openai",
            model="gpt-3.5-turbo-instruct",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"unsupported LLM: '{model}'")

    if model.startswith("Llama"):
        # result = Llama2Chat(llm=result)
        pass
    elif model.startswith("Mixtral"):
        pass
        llm = Mixtral(llm=llm)  # type: ignore

    if json_mode:  # NOT TESTED
        # see also https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html

        if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatGroq):
            llm = llm.bind(response_format={"type": "json_object"})
        else:
            raise ValueError(f"json_mode  not supported for {type(llm)}")

    return llm


def llm_getter(temp=0) -> Runnable:
    """
    Define a configurable LLM runnable

    https://python.langchain.com/docs/expression_language/primitives/configure/#with-llms-1
    """
    default = get_config("llm", "default_model")
    alternatives = {
        llm: llm_factory(llm, temperature=temp) for llm in KNOWN_LLM if llm != default
    }
    return (
        llm_factory()  # select default LLM
        .with_fallbacks([llm_factory("gpt_35_openai", temp)])  # To be changed
        .configurable_alternatives(
            ConfigurableField(id="llm"),
            default_key=default,
            prefix_keys=False,
            **alternatives,
        )
    )


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

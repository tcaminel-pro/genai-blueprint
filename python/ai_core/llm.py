"""
LLM models factory

"""

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

from lunary import LunaryCallbackHandler
from python.config import get_config


class LmmInfo(BaseModel):
    name: str
    model: str
    alt_name: list[str] = []


MAX_TOKEN = 2048

KNOWN_LLM = {
    "gpt_3_openai",
    "gpt_3_edenai",
    "llama2_70_deepinfra",
    "llama2_70_groq",
    "Llama3_70_groq",
    "mixtral_7x8_deepinfra",
    "mixtral_7x8_groq",
}


def llm_factory(model: str | None = None, temperature=0) -> BaseLanguageModel:
    if model is None:
        model = get_config("llm", "default_model")

    if model == "gpt_3_openai":
        result = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=temperature,
            model_kwargs={"seed": 42},  # Not sure that works
        )

    elif model == "llama2_70_deepinfra":
        result = DeepInfra(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": MAX_TOKEN,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )

    elif model == "mixtral_7x8_deepinfra":
        result = DeepInfra(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": MAX_TOKEN,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )

    elif model == "llama2_70_groq":
        result = ChatGroq(
            model="lLama2-70b-4096",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )
    elif model == "Llama3_70_groq":
        result = ChatGroq(
            model="lLama3-70b-8192",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )
    elif model == "mixtral_7x8_groq":
        result = ChatGroq(
            model="Mixtral-8x7b-32768",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )

    elif model == "gpt_3_edenai":
        result = EdenAI(
            feature="text",
            provider="openai",
            model="gpt-3.5-turbo-instruct",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )

    else:
        raise ValueError(f"unsupported LLM: '{model}'")

    if model.startswith("Llama"):
        # result = Llama2Chat(llm=result)
        pass
    elif model.startswith("Mixtral"):
        pass
        # result = Mixtral(llm=result)  # type: ignore

    return result


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
        .with_fallbacks([llm_factory("gpt_3_openai", temp)])  # To be changed
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

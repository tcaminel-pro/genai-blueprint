"""
LLM models factory

"""

from functools import cache

from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat, Mixtral
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

from langchain_community.llms.deepinfra import DeepInfra
from langchain_groq import ChatGroq

from lunary import LunaryCallbackHandler
from python.config import get_config


MAX_TOKEN = 2048


@cache
def get_llm(model: str | None = None, temperature=0) -> BaseLanguageModel:
    if model is None:
        model = get_config("llm", "default_model")

    if model in ["gpt-3.5-turbo_OpenAI", "gpt-3.5"]:
        result = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=temperature,
            model_kwargs={"seed": 42},  # Not sure that works
        )

    elif model == "Llama2_70B_DeepInfra":
        result = DeepInfra(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": MAX_TOKEN,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )

    elif model == "Mixtral_7x8B_DeepInfra":
        result = DeepInfra(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": MAX_TOKEN,
                "repetition_penalty": 1.3,
                "stop": ["STOP_TOKEN"],
            },
        )

    elif model == "Llama2_70B_Groq":
        result = ChatGroq(
            model="lLama2-70b-4096",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )
    elif model == "Mixtral_7x8B_Groq":
        result = ChatGroq(
            model="Mixtral-8x7b-32768",
            temperature=temperature,
            max_tokens=MAX_TOKEN,
        )

    else:
        raise ValueError(f"unsupported LLM: '{model}'")

    if model.startswith("Llama"):
        result = Llama2Chat(llm=result)  # type: ignore
    elif model.startswith("Mixtral"):
        result = Mixtral(llm=result)  # type: ignore

    set_cache()
    return result


def set_cache():
    # Set LangChain prompt cache
    cache = get_config("llm", "cache")
    if cache == "memory":
        set_llm_cache(InMemoryCache())
    elif cache == "sqlite":
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    else:
        raise ValueError(
            "incorrect [llm]/cache config. Should be 'memory' or 'sqlite' "
        )


llm_monitor_handler = LunaryCallbackHandler()  # Not used yet

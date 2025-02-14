"""The usual "tell me a joke" LLM call."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.ai_core.chain_registry import (
    Example,
    RunnableItem,
    register_runnable,
)
from src.ai_core.llm import get_llm
from src.ai_core.prompts import def_prompt

load_dotenv(verbose=True)


def get_chain(config: dict) -> Runnable:
    simple_prompt = """Tell me a joke on {topic}"""
    chain = {"topic": RunnablePassthrough()} | def_prompt(user=simple_prompt) | get_llm() | StrOutputParser()
    return chain


# Register the chain
register_runnable(
    RunnableItem(
        tag="Agent",
        name="Joke",
        runnable=get_chain,
        examples=[Example(query=["Beaver"])],
    )
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from python.ai_core.chain_registry import (
    RunnableItem,
    register_runnable,
)
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt

user_prompt = """Tell me a joke on {topic}"""
joke_chain = (
    {"topic": RunnablePassthrough()}
    | def_prompt(user=user_prompt)
    | get_llm()
    | StrOutputParser()
)

register_runnable(
    RunnableItem(
        tag="Agent",
        name="Joke",
        runnable=joke_chain,
        examples=["French"],
    )
)
"""
Entry point for the REST API

"""

from dotenv import load_dotenv
from fastapi import FastAPI
from genai_tk.core.chain_registry import ChainRegistry
from langchain_openai import ChatOpenAI
from langserve import add_routes

"""The usual "tell me a joke" LLM call."""

from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

load_dotenv(verbose=True)


def get_chain() -> Runnable:
    simple_prompt = """Tell me a joke on {topic}"""
    chain = (
        {"input": RunnablePassthrough(), "root": RunnablePassthrough()}
        | def_prompt(user=simple_prompt)
        | get_llm()
        | StrOutputParser()
    )
    return chain


load_dotenv(verbose=True)

chain_registry = ChainRegistry.instance()
ChainRegistry.load_modules()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


add_routes(app, {"input": RunnablePassthrough()} | def_prompt(user="tell ma a joke") | get_llm(), path="/joke")


add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo-0125"),
    path="/openai",
)

for runnable in chain_registry.get_runnable_list():
    add_routes(app, runnable.get(), path="/" + runnable.name.lower())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

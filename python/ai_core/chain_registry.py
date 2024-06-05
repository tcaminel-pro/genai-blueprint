"""
A registry for Langchain Runnables 
"""

import importlib
from pathlib import Path
from typing import Any, Callable, Tuple

from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from python.config import get_config_list, get_config_str


class Example(BaseModel):
    """Define prompt examples. Can also point to a file (for RAG)"""

    query: list[str]
    path: Path | None = None
    # ext: str | None = None


class RunnableItem(BaseModel):
    """
    A LangChain 'Runnable' encapsulation to be used in a Runnable registry.

    The runnable can be given in 3 forms:
        - As an object of class Runnable
        - As a function that returns un object of class Runnable that has a string as parameter
        - As a tuple, with a key (str) and  a function that returns un object of class Runnable that has a dict with given key as parameter

    Additionally, it's possible to attach to the Runnable some information to create de demo:
        - Examples of prompts
        - Diagram
    """

    tag: str
    name: str
    runnable: Runnable | Tuple[str, Callable[[dict[str, Any]], Runnable]] | Callable[
        [dict[str, Any]], Runnable
    ]  # Either a Runnable, or ...
    examples: list[Example] = []
    diagram: str | None = None

    def invoke(self, input: str, conf: dict[str, Any]) -> Any:
        runnable = self.get(conf)
        # is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=conf)
        result = runnable.invoke(input)
        return result

    def get(self, conf={"llm": None}) -> Runnable:
        if isinstance(self.runnable, Runnable):
            runnable = self.runnable
        elif isinstance(self.runnable, Callable):
            runnable = self.runnable(conf)
        elif isinstance(self.runnable, Tuple):
            key, func = self.runnable
            func_runnable = _to_key_param_callable(key, func)
            runnable = func_runnable(conf)
        else:
            raise Exception("unknown or ill-formatted Runnable")
        # debug(self.runnable, runnable)
        return runnable

    class Config:
        arbitrary_types_allowed = True


# Global registry
_registry: list[RunnableItem] = []


def register_runnable(r: RunnableItem):
    _registry.append(r)


def get_runnable_registry():
    return _registry


def find_runnable(name: str) -> RunnableItem | None:
    # return the 'RunnableItem' from its name. Case is ignored
    return next(
        (x for x in _registry if x.name.strip().lower() == name.strip().lower()), None
    )


def _to_key_param_callable(
    key: str, function: Callable[[dict[str, Any]], Runnable]
) -> Callable[[Any], Runnable]:
    """
    Tricky function :
    Take a function having a config parameter and returning a Runnable whose input is a string,
    and return a function where the same Runnable takes a dict instead of the string.
    """
    return lambda conf: RunnableLambda(lambda x: {key: x}) | function(conf)


def load_modules_with_chains():
    """
    Import modules given in configuration that contains Chains to be registered
    """
    path = get_config_str("chains", "path")
    modules = get_config_list("chains", "modules")
    assert Path(path).exists

    for module in modules:
        importlib.import_module(f"{path}.{module}")

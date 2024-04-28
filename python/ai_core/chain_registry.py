from typing import Any, Callable

from devtools import debug  # ignore
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class RunnableItem(BaseModel):
    tag: str
    name: str
    runnable: Runnable | Callable[[dict[str, Any]], Runnable]
    examples: list[str] = []
    key: str | None = None

    def invoke(self, input: str, conf: dict[str, Any]) -> Any:
        runnable = self.get_runnable(conf)
        # is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=conf)

        # try to determine the name of the key for the query.
        # does not work for Agent Executor, but there might be a way.
        # input_schema_props = runnable.input_schema().schema().get("properties")
        # key = None
        # if input_schema_props:
        #     input_keys = input_schema_props.keys()
        #     assert len(input_keys) == 1
        #     key = next(iter(input_keys))

        if self.key:
            result = runnable.invoke({self.key: input})
        else:
            result = runnable.invoke(input)
        return result

    def get_runnable(self, conf={"llm": None}) -> Runnable:
        if isinstance(self.runnable, Runnable):
            runnable = self.runnable
        elif isinstance(self.runnable, Callable):
            runnable = self.runnable(conf)
        else:
            raise Exception("unknown Runnable")
        return runnable

    class Config:
        arbitrary_types_allowed = True


_registry: list[RunnableItem] = []


def register_runnable(r: RunnableItem):
    _registry.append(r)


def get_runnable_registry():
    return _registry


def find_runnable(name: str) -> RunnableItem | None:
    return next((x for x in _registry if x.name == name), None)


def to_key_param_callable(
    key: str, function: Callable[[dict[str, Any]], Runnable]
) -> Callable[[Any], Runnable]:
    """
    Take a function taking a config parameter and returning a Runnable whose input is a string,
    and return a function where the same Runnable takes a dict instead.
    """
    return lambda conf: RunnableLambda(lambda x: {key: x}) | function(conf)

from typing import Any, Callable
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain.agents import create_tool_calling_agent, AgentExecutor

from python.config import get_config


class RunnableItem(BaseModel):
    category: str
    description: str
    runnable: Runnable | Callable[[dict[str, Any]], Runnable]
    examples: list[str] = []

    def invoke(self, input: str, config: dict[str, Any]) -> Any:
        if isinstance(self.runnable, Runnable):
            runnable = self.runnable
        elif isinstance(self.runnable, Callable):
            runnable = self.runnable(config)
        else:
            raise Exception("unknown Runnable")
        is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=config)
        if is_agent:
            print("==> Agent Executor")
            result = runnable.invoke({"input": input})
        else:
            result = runnable.invoke(input)
        return result

    class Config:
        arbitrary_types_allowed = True


_registry: list[RunnableItem] = []


def register_runnable(
    category: str,
    description: str,
    runnable: Runnable | Callable[[dict[str, Any]], Runnable],
    examples: list[str] = [],
):
    r = RunnableItem(
        category=category, description=description, runnable=runnable, examples=examples
    )
    _registry.append(r)


def get_runnable_registry():
    return _registry

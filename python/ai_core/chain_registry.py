from typing import Any, Callable
from pydantic import BaseModel
from langchain_core.runnables import Runnable


class RunnableItem(BaseModel):
    tag: str
    name: str
    #    runnable: Runnable | Callable[[dict[str, Any]], Runnable]
    runnable: Runnable | Callable
    examples: list[str] = []
    key: str | None = None

    def invoke(self, input: str, config: dict[str, Any]) -> Any:
        if isinstance(self.runnable, Runnable):
            runnable = self.runnable
        elif isinstance(self.runnable, Callable):
            runnable = self.runnable(config)
        else:
            raise Exception("unknown Runnable")
        # is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=config)
        if self.key:
            result = runnable.invoke({self.key: input})
        else:
            result = runnable.invoke(input)
        return result

    class Config:
        arbitrary_types_allowed = True


_registry: list[RunnableItem] = []


def register_runnable(r: RunnableItem):
    _registry.append(r)


def get_runnable_registry():
    return _registry


def find_runnable(name: str) -> RunnableItem | None:
    return next((x for x in _registry if x.name == name), None)

from pydantic import BaseModel
from langchain_core.runnables import Runnable


class RunnableDescription(BaseModel):
    category: str
    description: str
    runnable: Runnable
    examples: list[str] = []

    class Config:
        arbitrary_types_allowed = True


_registry: list[RunnableDescription] = []


def register_runnable(
    category: str, description: str, runnable: Runnable, examples: list[str] = []
):
    r = RunnableDescription(
        category=category, description=description, runnable=runnable, examples=examples
    )
    _registry.append(r)


def get_runnable_registry():
    return _registry

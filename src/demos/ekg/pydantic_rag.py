from typing import TypeVar

from pydantic import BaseModel, PostgresDsn

from src.ai_core.vector_store import VectorStoreFactory

T = TypeVar("T", bound=BaseModel)

"""


"""
class PydanticRag(BaseModel):
    model_definition: str
    postres_url: PostgresDsn

    _vector_store_fact: VectorStoreFactory

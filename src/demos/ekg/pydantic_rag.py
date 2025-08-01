from typing import TypeVar

from pydantic import BaseModel, PostgresDsn

from src.ai_core.vector_store import VectorStoreFactory

T = TypeVar("T", bound=BaseModel)

"""
Complete that class that analyse / query documents having same structure with an LLM.
1 / Load a YAML string and create a Pydantic class, using    class YamlToPydantic().create_class_from_yaml(...)
2/ Have a method to analyse a given markdown document with an LLM and that Pydantic class, as done in process_markdown_batch 
3/ That method produce a Pydantic object. Calculate an embeddings (configurable) for each part of the obhect, as in pydantic_embeddings.py
4/ Store the embedddings in a PgVector vectorstore.  Use as additional metadata fields the name of the top class and the name of the field
5/ Add additional method do query that VectorStore

"""
class PydanticRag(BaseModel):
    model_definition: str
    postres_url: PostgresDsn

    _vector_store_fact: VectorStoreFactory

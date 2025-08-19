"""RAG system implementation for semantic search and querying of structured data.

This module provides the tool creation and query interface components for:
- Creating LangChain tools for semantic search
- Defining query schemas and filters
- Processing search results into structured outputs

Key Components:
    - PydanticRag: Main class handling tool creation and query processing
    - Vector search tool with configurable filters
    - Result processing and formatting
"""

from typing import Any, List, Optional, TypeVar

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from loguru import logger
from pydantic import BaseModel, Field

from src.ai_core.prompts import dedent_ws
from src.demos.ekg.extract_and_store import PydanticRagBase

T = TypeVar("T", bound=BaseModel)


class PydanticRag(PydanticRagBase):
    """RAG system extension for creating semantic search tools and processing queries.

    Inherits from PydanticRagBase and adds tool creation capabilities.
    Handles the query interface and result formatting aspects of the RAG system.
    """

    def get_top_class_description(self) -> str:
        """Return the description field associated with the top class in the schema."""
        return self.model_definition.get("schema", {}).get(self._top_class.__name__, {}).get("description", "")

    def get_key_description(self) -> str:
        """Return the description of the key field from the model definition."""
        key_path = self._key_field.split(".")
        current_schema = self.model_definition.get("schema", {})

        top_class_name = self.model_definition.get("top_class", "")
        if not top_class_name or top_class_name not in current_schema:
            return ""

        description = ""
        temp_schema = current_schema

        if "fields" in current_schema[top_class_name]:
            temp_schema = current_schema[top_class_name]["fields"]

        for i, key_part in enumerate(key_path):
            if key_part in temp_schema:
                field_def = temp_schema[key_part]

                if i == len(key_path) - 1:
                    description = field_def.get("description", "")
                    break

                if "type" in field_def:
                    field_type = field_def["type"]
                    if field_type.startswith("list[") and field_type.endswith("]"):
                        field_type = field_type[5:-1]

                    if field_type in current_schema:
                        if "fields" in current_schema[field_type]:
                            temp_schema = current_schema[field_type]["fields"]
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                break

        return description

    def create_vector_search_tool(self) -> BaseTool:
        """Create a LangChain BaseTool for searching the vector store."""
        _entity_id_name: str = self._key_field.split(".")[-1]
        _top_class_description: dict = self.get_top_class_fields()
        _entity_key_name: str = self._key_field.split(".")[-1]

        class _VectorSearchInput(BaseModel):
            """Input schema for the vector search tool."""

            query: str = Field(
                ...,
                description=dedent_ws(
                    """
                    The query to the semantic search vector store. 
                    Provide several variants of the request to improve the semantic matching 
                    (ex: broaden, examples,...) """
                ),
            )
            selected_sections: List[str] = Field(
                ...,
                description=dedent_ws(
                    f"""
                    List of sections relevant for the query.\n
                    Allowed section name SHOULD BE in that list: \n - {"\n- ".join([f"'{k}' ({v})" for k, v in self.get_top_class_fields().items()])} \n
                    Select only the most relevant sections (maximum 2)"""
                ),
            )
            entity_keys: list[str] = Field(
                ...,
                description=dedent_ws(
                    f"""
                    List of '{_entity_id_name}' mentionned in the discussion and whose user is talking about. 
                    (for example the one returned by a search query and whose user wants more details).
                    Return empty list if no '{_entity_id_name}' has been mentioned, or if the user request is not 
                    related to previously returned  {_entity_id_name}"""
                ),
            )

        class _VectorSearchTool(BaseTool):
            """Tool for searching the vector store for semantic matches."""

            name: str = f"{self.get_top_class().__name__}_retriever"
            description: str = dedent_ws(
                f"""
                Retrieve information related to documents described as '{self.get_top_class_description()}.
                Each document is related to a unique id '{_entity_id_name}', with is typically a  {self.get_key_description()}.
                Argument are:
                - expanded query.  
                - selected_sections: a list of section names that best match the query. Select several if you are not sure.
                - entity_keys:  list of '{_entity_id_name}' mentionned in the context / discussion..
                """
            )
            args_schema: Optional[ArgsSchema] = _VectorSearchInput

            def model_post_init(self, __context: Any) -> None:
                self._vector_store = PydanticRag.get_vector_store_factory().get()

            def _run(self, query: str, selected_sections: List[str], entity_keys: list[str] = []) -> str:
                """Execute search against the vector store."""
                allowed = set(_top_class_description.keys())
                invalid = [f for f in selected_sections if f not in allowed]
                if invalid:
                    logger.warning(f"Removing invalid section: {invalid}")
                    selected_sections = [f for f in selected_sections if f in allowed]
                section_filter = {"field_name": {"$in": selected_sections}} if selected_sections else {}
                entity_filter = {"entity_id": {"$in": entity_keys}} if entity_keys else {}

                filter_dict = {"$and": [section_filter, entity_filter]} if entity_filter else section_filter
                docs = self._vector_store.similarity_search(query, k=20, filter=filter_dict)

                if not docs:
                    return "No information found"

                entity_id = docs[0].metadata.get("entity_id", "unknown")
                fields_dict = {}
                for doc in docs:
                    field_name = doc.metadata.get("field_name", "")
                    fields_dict.setdefault(field_name, []).append(doc)

                result_parts = [f"# {_entity_key_name}: {entity_id}"]
                field_order = list(_top_class_description.keys())
                sorted_fields = sorted(
                    fields_dict.keys(), key=lambda x: field_order.index(x) if x in field_order else len(field_order)
                )

                for field_name in sorted_fields:
                    result_parts.append(f"## {field_name}")
                    for doc in fields_dict[field_name]:
                        content = doc.page_content.replace("#", "###")
                        result_parts.append(f"\n{content}")

                return "\n".join(result_parts)

        return _VectorSearchTool()

    def get_top_class_fields(self) -> dict:
        return {
            field_name: getattr(field_info, "description", "")
            for field_name, field_info in self._top_class.model_fields.items()
            if field_name != "source"
        }

    def query_vectorstore(self, query: str, k: int = 4, filter: dict = {}) -> List[Document]:
        """Search the vector store for similar field data."""
        vector_store = self.vector_store_factory.get()
        return vector_store.similarity_search(query, k=k, filter=filter)

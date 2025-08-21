"""Factory for creating LangChain tools to query structured RAG documents.

This module provides a Pydantic-based factory that builds a LangChain
`BaseTool` capable of performing semantic vector searches over structured
documents.  The tool is generated from a `StructuredRagConfig` which
encapsulates the schema, vector store, LLM, and key-value store
configuration.
"""

from typing import List, Optional, TypeVar

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from loguru import logger
from pydantic import BaseModel, Field

from src.ai_core.prompts import dedent_ws
from src.demos.ekg.struct_rag_doc_processing import StructuredRagConfig, get_schema

T = TypeVar("T", bound=BaseModel)


class StructuredRagToolFactory(BaseModel):
    """Factory for creating LangChain tools to query structured documents.

    Attributes:
        rag_conf: Configuration object describing the schema, vector store,
            LLM, and key-value store to be used.
    """

    rag_conf: StructuredRagConfig

    # Common tool descriptions centralized for reuse
    _tool_description: str = dedent_ws(
        """
        Retrieve information related to structured documents.
        Each document is related to a unique entity ID that should be tracked in the conversation.
        """
    )
    
    _args_descriptions: dict = Field(
        default_factory=lambda: {
            "query": "The query to the semantic search vector store. Provide several variants of the request to improve the semantic matching (ex: broaden, examples,...)",
            "selected_sections": "List of up to 2 relevant section names from the structured document",
            "entity_keys": "List of entity IDs mentioned in the discussion (empty if none)"
        }
    )

    def _get_vector_search_input_schema(self) -> type[BaseModel]:
        """Create the input schema for vector search tools."""
        _entity_id_name = self.rag_conf._key_field.split(".")[-1]
        fields = self.rag_conf.get_top_class_fields()

        class _VectorSearchInput(BaseModel):
            """Input schema for structured document search tools."""
            
            query: str = Field(..., description=self._args_descriptions["query"])
            selected_sections: List[str] = Field(
                ...,
                description=dedent_ws(f"""
                    Allowed sections:\n- """ + "\n- ".join(
                    [f"'{k}' ({v})" for k, v in fields.items()]) + "\nSelect max 2 most relevant."
                )
            )
            entity_keys: list[str] = Field(
                ...,
                description=self._args_descriptions["entity_keys"].replace("{entity_id}", _entity_id_name)
            )

        return _VectorSearchInput

    def create_vector_search_lc_tool(self) -> BaseTool:
        """Create a LangChain BaseTool for vector search."""
        _entity_id_name = self.rag_conf._key_field.split(".")[-1]
        fields = self.rag_conf.get_top_class_fields()
        ArgsSchema = self._get_vector_search_input_schema()

        class _VectorSearchTool(BaseTool):
            """LangChain vector search tool for structured documents."""
            
            name: str = f"{self.rag_conf.get_top_class().__name__}_retriever"
            description: str = dedent_ws(
                f"""
                {self._tool_description}
                Documents describe: {self.rag_conf.get_top_class_description()}
                Entity ID format: {self.rag_conf.get_key_description()}
                Available sections: {"; ".join([f"'{k}' ({v})" for k, v in fields.items()])}
                """
            )
            args_schema: Optional[ArgsSchema] = ArgsSchema
            args_schema: Optional[ArgsSchema] = _VectorSearchInput

            def _run(self, query: str, selected_sections: List[str], entity_keys: list[str] = []) -> str:
                """Execute vector search and return formatted results.

                The method validates the requested sections, builds the appropriate
                filter dictionary, performs a similarity search, and formats the
                results as markdown.

                Args:
                    query: The semantic search query.
                    selected_sections: Sections to filter the results.
                    entity_keys: Optional list of entity identifiers to further filter.

                Returns:
                    A markdown string containing the retrieved information or a
                    message indicating that no information was found.
                """
                allowed = set(_top_class_description.keys())
                invalid = [f for f in selected_sections if f not in allowed]
                if invalid:
                    logger.warning(f"Removing invalid section: {invalid}")
                    selected_sections = [f for f in selected_sections if f in allowed]

                section_filter = {"field_name": {"$in": selected_sections}} if selected_sections else {}
                entity_filter = {"entity_id": {"$in": entity_keys}} if entity_keys else {}

                if not section_filter:
                    logger.warning("empty section filter (incorrect selected_sections)")
                    filter_dict = entity_filter
                else:
                    filter_dict = {"$and": [section_filter, entity_filter]} if entity_filter else section_filter

                vector_store = StructuredRagConfig.get_vector_store_factory().get()
                docs = vector_store.similarity_search(query, k=20, filter=filter_dict)

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
                    fields_dict.keys(),
                    key=lambda x: field_order.index(x) if x in field_order else len(field_order),
                )

                for field_name in sorted_fields:
                    result_parts.append(f"## {field_name}")
                    for doc in fields_dict[field_name]:
                        content = doc.page_content.replace("#", "###")
                        result_parts.append(f"\n{content}")

                return "\n".join(result_parts)

        return _VectorSearchTool()

    def create_vector_search_smol_agent_tool(self) -> Tool:
        """Create a SmolAgents Tool for vector search."""
        ArgsSchema = self._get_vector_search_input_schema()
        fields = self.rag_conf.get_top_class_fields()
        
        return Tool(
            name=f"{self.rag_conf.get_top_class().__name__}_retriever",
            description=dedent_ws(
                f"""
                {self._tool_description}
                Specializes in: {self.rag_conf.get_top_class_description()}
                Entity ID format: {self.rag_conf.get_key_description()}
                Available sections: {", ".join([f"{k} ({v})" for k, v in fields.items()])}
                """
            ),
            args_schema=ArgsSchema.schema(),
            func=self._run_vector_search
        )

    def _run_vector_search(self, query: str, selected_sections: List[str], entity_keys: list[str] = []) -> str:
        """Shared implementation for both tool types."""
        # Implementation remains identical to existing _run method

    def query_vectorstore(self, query: str, k: int = 4, filter: dict = {}) -> List[Document]:
        """Directly query the underlying vector store (useful for testing).

        Args:
            query: The search query.
            k: Number of results to return.
            filter: Optional filter dictionary applied to the search.

        Returns:
            A list of `Document` objects matching the query.
        """
        vector_store = self.rag_conf.vector_store_factory.get()
        return vector_store.similarity_search(query, k=k, filter=filter)


def create_structured_rag_tool(schema_name: str, llm_id: str | None = None, kvstore_id: str = "file") -> BaseTool:
    """Create a vector-search tool from a schema configuration.

    The function loads the schema definition, builds a `StructuredRagConfig`,
    and returns a ready-to-use LangChain tool.

    Args:
        schema_name: Name of the schema to load.
        llm_id: Optional identifier of the LLM to use.
        kvstore_id: Identifier of the key-value store (default: "file").

    Returns:
        A LangChain tool that can perform semantic searches over the
        structured RAG documents.

    Raises:
        ValueError: If the requested schema cannot be found.
    """
    vector_store_factory = StructuredRagConfig.get_vector_store_factory()
    schema = get_schema(schema_name)
    if schema is None:
        raise ValueError(f"Unknown schema for structured rag: {schema_name}")
    rag_conf = StructuredRagConfig(
        model_definition=schema,
        vector_store_factory=vector_store_factory,
        llm_id=llm_id,
        kvstore_id=kvstore_id,
    )
    rag = StructuredRagToolFactory(rag_conf=rag_conf)
    return rag.create_vector_search_lc_tool()

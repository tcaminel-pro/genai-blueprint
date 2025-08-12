from typing import Any, List, Optional, Type, TypeVar

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from markpickle import dumps
from pydantic import BaseModel, Field, PrivateAttr

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.llm_factory import get_llm
from src.ai_core.prompts import def_prompt
from src.utils.pydantic.dyn_model_factory import PydanticModelFactory
from src.utils.pydantic.kv_store import load_object_from_kvstore, save_object_to_kvstore

T = TypeVar("T", bound=BaseModel)


class PydanticRag(BaseModel):
    """RAG system for analyzing and querying structured documents using Pydantic models and embeddings.

    Attributes:
        model_definition: YAML string defining the Pydantic model
        postgres_url: PostgreSQL connection URL
        embeddings_id: ID for embeddings model
        llm_id: ID for LLM used in analysis
        collection_name: Name of vector store collection (default: "pydantic_fields")
    """

    model_definition: dict
    postgres_url: str
    embeddings_id: str
    llm_id: str | None
    kv_store_id: str | None = None
    collection_name: str = "pydantic_fields"

    _top_class: Type[BaseModel] = PrivateAttr()
    _vector_store: VectorStore = PrivateAttr()
    _key_field: str = PrivateAttr()
    _llm: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize with configuration and create required components."""
        """Create Pydantic class from YAML definition."""
        required_keys = ["schema", "key", "top_class"]
        for k in required_keys:
            if self.model_definition.get(k) is None:
                raise ValueError("Model should have key '{k}'")
        converter = PydanticModelFactory()
        self._top_class = converter.create_class_from_dict(
            self.model_definition["schema"], self.model_definition["top_class"]
        )
        self._key_field = self.model_definition["key"]
        self._llm = get_llm(llm_id=self.llm_id, streaming=False, temperature=0.0)
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize the vector store with embeddings model."""
        from src.ai_core.vector_store_factory import VectorStoreFactory

        self._vector_store = VectorStoreFactory(
            id="PgVector",
            embeddings_factory=EmbeddingsFactory(embeddings_id=self.embeddings_id),
            config={
                "postgres_url": self.postgres_url,
                "hybrid_search": True,
                "metadata_columns": [
                    {"name": "document_id", "data_type": "VARCHAR"},
                    {"name": "field_name", "data_type": "VARCHAR"},
                    {"name": "model_class", "data_type": "VARCHAR"},
                ],
            },
            table_name_prefix=self.collection_name,
        ).get()

    def analyze_document(self, document_id: str, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data."""

        # self._document_id = document_id
        analyzed_doc: BaseModel | None = None
        if self.kv_store_id:
            analyzed_doc = load_object_from_kvstore(self.get_top_class(), key=document_id, kv_store_id=self.kv_store_id)
        if not analyzed_doc:
            system = f"""
                Extract structured information from the document below into {self._top_class.__name__} schema. 
                Answer with a JSON document. Avoid explanations or extra text."""
            user = "Document to analyze:\n---\n{input}\n---"
            chain = def_prompt(system=system, user=user) | self._llm.with_structured_output(self._top_class)
            analyzed_doc = chain.invoke({"input": markdown})
            assert analyzed_doc
            analyzed_doc.__setattr__("document_id", document_id)
            if self.kv_store_id:
                save_object_to_kvstore(document_id, analyzed_doc, kv_store_id="file")

        return analyzed_doc

    def get_top_class(self) -> type[BaseModel]:
        return self._top_class

    def get_key(self, obj: BaseModel) -> str:
        """Extract the actual key value from a model instance using dotted notation from key field definition."""
        key_path = self._key_field.split(".")
        current_value = obj
        # Navigate through the path
        for key_part in key_path:
            if isinstance(current_value, dict):
                current_value = current_value.get(key_part)
            else:
                current_value = getattr(current_value, key_part, None)
            if current_value is None and key_part == key_path[0]:  # the fiest element might be the current field
                current_value = obj
        if current_value is None:
            raise ValueError(f"incorrect key {self._key_field}")
        return str(current_value)

    def store_chunks(self, chunks: list[Document]) -> None:
        """Store a document's field embeddings in vector store."""
        self._vector_store.add_documents(chunks)

    def query_vectorstore(self, query: str, k: int = 4, filter: dict = {}) -> List[Document]:
        """Search the vector store for similar field data."""
        return self._vector_store.similarity_search(query, k=k, filter=filter)

    def chunck(self, model_instance: BaseModel) -> list[Document]:
        """Generate LangChain Document objects for each field in a Pydantic model.

        Creates Document objects with YAML content as page_content and metadata
        including field information.

        Args:
            model_instance: An instance of a Pydantic model

        """
        documents = []
        model_data = model_instance.model_dump()
        document_id = getattr(model_instance, "document_id", None)

        for field_name, field_value in model_data.items():
            if field_value is None:
                continue
            field_info = type(model_instance).model_fields.get(field_name)
            if field_info is None:
                continue
            page_content = f"{dumps(field_value)}"  # serialize as Markdown
            field_doc = Document(
                page_content=page_content,
                metadata={
                    "field_name": field_name,
                    "model_class": model_instance.__class__.__name__,
                    "description": getattr(field_info, "description", "") or "",
                    "document_id": document_id,
                },
            )
            documents.append(field_doc)
        return documents

    def create_vector_search_tool(self) -> BaseTool:
        """Create a LangChain BaseTool for searching the vector store.

        Returns:
            A BaseTool that can search the vector store for semantic matches.
        """
        from pydantic import BaseModel, Field

        class VectorSearchInput(BaseModel):
            """Input schema for the vector search tool."""

            query: str = Field(..., description="The search query to find semantically similar documents")
            fields: Optional[List[str]] = Field(
                None, description="Optional list of field names to limit the search to specific fields in the documents"
            )

        class VectorSearchTool(BaseTool):
            """Tool for searching the vector store for semantic matches."""

            name: str = "vector_search"
            description: str = self._create_tool_description()
            args_schema: Type[BaseModel] = VectorSearchInput

            def _run(self, query: str, fields: Optional[List[str]] = None) -> List[Document]:
                """Execute search against the vector store.

                Args:
                    query: The user's search query
                    fields: Optional list of field names to filter results

                Returns:
                    List of matching documents
                """
                filter_dict = {}
                if fields:
                    # Create filter for specific fields
                    filter_dict = {"field_name": {"$in": fields}}

                return self._vector_store.similarity_search(query, k=4, filter=filter_dict)

        return VectorSearchTool()

    def _create_tool_description(self) -> str:
        """Create a description for the tool based on top-level field descriptions."""
        if not hasattr(self._top_class, "model_fields"):
            return "Search documents by semantic similarity across all fields"

        descriptions = []
        for field_name, field_info in self._top_class.model_fields.items():
            description = getattr(field_info, "description", "")
            if description:
                descriptions.append(f"{field_name}: {description}")

        if not descriptions:
            return "Search documents by semantic similarity across all fields"

        return "Search documents by semantic similarity. Fields: " + "; ".join(descriptions)

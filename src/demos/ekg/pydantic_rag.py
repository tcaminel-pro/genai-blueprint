from typing import Any, List, Type, TypeVar
from uuid import uuid4

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, PostgresDsn, PrivateAttr

from src.ai_core.embeddings import get_embeddings
from src.ai_core.llm import get_llm
from src.utils.pydantic.yaml_to_pydantic import YamlToPydantic

from .pydantic_embeddings import generate_field_documents

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

    model_definition: str
    postgres_url: PostgresDsn
    embeddings_id: str
    llm_id: str
    collection_name: str = "pydantic_fields"

    _top_class: Type[BaseModel] = PrivateAttr()
    _vector_store: VectorStore = PrivateAttr()
    _llm: Any = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        """Initialize with configuration and create required components."""
        super().__init__(**data)
        self.create_pydantic_class()
        self._init_llm()
        self._init_vector_store()

    def create_pydantic_class(self) -> None:
        """Create Pydantic class from YAML definition."""
        converter = YamlToPydantic()
        self._top_class = converter.create_class_from_yaml(self.model_definition)

    def _init_llm(self) -> None:
        """Initialize the language model for document analysis."""
        self._llm = get_llm(llm_id=self.llm_id, streaming=False)

    def _init_vector_store(self) -> None:
        """Initialize the vector store with embeddings model."""
        embeddings = get_embeddings(self.embeddings_id)
        self._vector_store = PGVector(
            connection_string=self.postgres_url.unicode_string(),
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )

    def analyze_document(self, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data (Placeholder)."""
        # TODO: Implement analysis using LLM conversion to structured data
        # This would involve converting markdown to the top_class format
        # Currently returns a default instance as placeholder
        return self._top_class()

    def analyze_and_store(self, markdown: str) -> None:
        """Analyze markdown and store field embeddings in vector store."""
        document = self.analyze_document(markdown)
        self.store_document(document)

    def store_document(self, document: BaseModel) -> None:
        """Store a document's field embeddings in vector store."""
        field_docs = generate_field_documents(document)
        doc_id = str(uuid4())
        for doc in field_docs:
            doc.metadata["document_id"] = doc_id
            doc.metadata["field_name"] = doc.metadata.get("field_name", "")
            doc.metadata["model_class"] = type(document).__name__
        if field_docs:
            self._vector_store.add_documents(field_docs)

    def query_vectorstore(self, query: str, k: int = 4) -> List[Document]:
        """Search the vector store for similar field data."""
        return self._vector_store.similarity_search(query, k=k)

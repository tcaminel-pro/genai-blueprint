from typing import Any, List, Type, TypeVar
from uuid import uuid4

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, PostgresDsn, PrivateAttr

from src.ai_core.embeddings import get_embeddings
from src.ai_core.llm import get_llm
from src.ai_core.prompts import def_prompt
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
        # Use temperature=0.0 for structured extraction
        self._llm = get_llm(llm_id=self.llm_id, streaming=False, temperature=0.0)

    def _init_vector_store(self) -> None:
        """Initialize the vector store with embeddings model."""
        embeddings = get_embeddings(self.embeddings_id)
        from src.ai_core.vector_store import VectorStoreFactory

        # Configure vector store with custom metadata for fields
        self._vector_store = VectorStoreFactory(
            id="pgvector",
            embeddings=embeddings,
            connection=self.postgres_url.unicode_string(),
            collection_name=self.collection_name,
            metadata_columns=["document_id", "field_name", "model_class"],
        ).get()

    def analyze_document(self, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data."""
        # System prompt for extraction
        system = f"Extract structured information from the document below into {self._top_class.__name__} schema. Answer with a JSON document. Avoid explanations or extra text."

        # User prompt with markdown content
        user = "Document to analyze:\n---\n{input}\n---"

        # Create chain that combines prompt and LLM with structured output
        chain = def_prompt(system=system, user=user) | self._llm.with_structured_output(self._top_class)

        return chain.invoke({"input": markdown})

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

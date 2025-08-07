from typing import Any, List, Type, TypeVar
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, PostgresDsn, PrivateAttr

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.llm_factory import get_llm
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

    def model_post_init(self, __context: Any) -> None:
        """Initialize with configuration and create required components."""
        """Create Pydantic class from YAML definition."""
        converter = YamlToPydantic()
        self._top_class = converter.create_class_from_yaml(self.model_definition)
        self._llm = get_llm(llm_id=self.llm_id, streaming=False, temperature=0.0)
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize the vector store with embeddings model."""
        from src.ai_core.vector_store_factory import VectorStoreFactory

        postgres_url = str(self.postgres_url)
        self._vector_store = VectorStoreFactory(
            id="PgVector",
            embeddings_factory=EmbeddingsFactory(embeddings_id=self.embeddings_id),
            config={
                "postgres_url": postgres_url,
                "hybrid_search": True,
                "metadata_columns": [
                    {"name": "document_id", "data_type": "VARCHAR"},
                    {"name": "field_name", "data_type": "VARCHAR"},
                    {"name": "model_class", "data_type": "VARCHAR"},
                ],
            },
            table_name_prefix=self.collection_name,
        ).get()

    def analyze_document(self, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data."""
        system = f"""
            Extract structured information from the document below into {self._top_class.__name__} schema. 
            Answer with a JSON document. Avoid explanations or extra text."""
        user = "Document to analyze:\n---\n{input}\n---"
        chain = def_prompt(system=system, user=user) | self._llm.with_structured_output(self._top_class)
        return chain.invoke({"input": markdown})

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


if __name__ == "__main__":
    """Minimal runnable demo.

    Prerequisites:
      1. Postgres running on postgresql://localhost:5432/postgres
      2. OPENAI_API_KEY in the environment
      3. uv add rich (for pretty printing)

    Run:
        uv run python -m src.demos.ekg.pydantic_rag
    """
    from rich import print

    EXAMPLE_YAML = """
    name: Person
    description: Basic contact card
    fields:
      name:
        type: str
        description: Full name of the person
      age:
        type: int
        description: Age in years
      email:
        type: str
        description: Primary e-mail address
    """

    rag = PydanticRag(
        model_definition=EXAMPLE_YAML,
        postgres_url="postgresql://localhost:5432/postgres",
        embeddings_id="openai",
        llm_id="gpt-4o",
    )

    # A tiny markdown document
    doc_text = """
    # Jane Doe
    Jane is 29 years old and can be reached at jane.doe@example.com.
    """

    # 1. Analyse → structured Pydantic object
    person = rag.analyze_document(doc_text)
    print("Structured result:", person)

    # 2. Index the document
    rag.store_document(person)
    print("Document stored.")

    # 3. Query the vector store
    hits = rag.query_vectorstore("e-mail address", k=2)
    print("Vector hits:", hits)

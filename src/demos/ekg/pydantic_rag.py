from typing import Any, List, Sequence, Type, TypeVar
from uuid import uuid4

import yaml
from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, PrivateAttr

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.llm_factory import get_llm
from src.ai_core.prompts import def_prompt
from src.utils.config_mngr import global_config
from src.utils.pydantic.yaml_to_pydantic import YamlToPydantic

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
    collection_name: str = "pydantic_fields"

    _top_class: Type[BaseModel] = PrivateAttr()
    _vector_store: VectorStore = PrivateAttr()
    _llm: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize with configuration and create required components."""
        """Create Pydantic class from YAML definition."""
        converter = YamlToPydantic()
        self._top_class = converter.create_class_from_dict(self.model_definition)
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

    def analyze_document(self, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data."""
        system = f"""
            Extract structured information from the document below into {self._top_class.__name__} schema. 
            Answer with a JSON document. Avoid explanations or extra text."""
        user = "Document to analyze:\n---\n{input}\n---"
        chain = def_prompt(system=system, user=user) | self._llm.with_structured_output(self._top_class)
        return chain.invoke({"input": markdown})

    def get_top_class(self) -> type[BaseModel]:
        return self._top_class

    def chunck(self, structured_doc: BaseModel) -> list[Document]:
        """Get Langchain Documents from structured dic"""
        transformer = PydanticFieldDocumentTransformer(include_null=False)
        field_docs = transformer.transform_documents([Document(
            page_content="",
            metadata={"structured_doc": structured_doc}
        )])
        doc_id = str(uuid4())
        for doc in field_docs:
            doc.metadata["document_id"] = doc_id
            doc.metadata["field_name"] = doc.metadata.get("field_name", "")
            doc.metadata["model_class"] = type(structured_doc).__name__
        return field_docs

    def store_chunks(self, chunks: list[Document]) -> None:
        """Store a document's field embeddings in vector store."""
        self._vector_store.add_documents(chunks)

    def query_vectorstore(self, query: str, k: int = 4) -> List[Document]:
        """Search the vector store for similar field data."""
        return self._vector_store.similarity_search(query, k=k)




class PydanticFieldDocumentTransformer(BaseDocumentTransformer, BaseModel):
    """Transform Pydantic model instances into LangChain Documents for each field.
    
    This transformer takes Pydantic model instances and converts them into individual
    Document objects, one for each field in the model. This enables fine-grained
    indexing and retrieval of structured data.
    
    Attributes:
        include_null: Whether to include fields with None values in the output
    """
    
    include_null: bool = False
    
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform Pydantic model instances into field-based Documents.
        
        Args:
            documents: Sequence of Documents containing Pydantic model instances
                      in their metadata under the key 'structured_doc'
            **kwargs: Additional arguments passed to the transformer
        
        Returns:
            Sequence of Documents, one for each field in the model instances
        """
        transformed_documents = []
        
        for doc in documents:
            model_instance = doc.metadata.get("structured_doc")
            if not isinstance(model_instance, BaseModel):
                continue
                
            for field_name, field_value in model_instance.model_dump().items():
                if field_value is None and not self.include_null:
                    continue

                field_info = type(model_instance).model_fields[field_name]

                yaml_content = yaml.dump(
                    {
                        "value": field_value,
                        "description": field_info.description,
                        "type": str(type(field_value).__name__) if field_value is not None else "None",
                    },
                    default_flow_style=False,
                    sort_keys=False,
                )

                field_doc = Document(
                    page_content=str(field_value),
                    metadata={
                        "field_name": field_name,
                        "model_class": model_instance.__class__.__name__,
                        "description": field_info.description or "",
                        "type": str(type(field_value).__name__) if field_value is not None else "None",
                    },
                )
                transformed_documents.append(field_doc)
        
        return transformed_documents
    
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform documents (delegates to sync version)."""
        return self.transform_documents(documents, **kwargs)


# Backward compatibility function
def generate_field_documents(
    model_instance: BaseModel,
    include_null: bool = False,
) -> List[Document]:
    """Generate LangChain Document objects for each field in a Pydantic model.

    Creates Document objects with YAML content as page_content and metadata
    including field information.

    Args:
        model_instance: An instance of a Pydantic model
        include_null: Whether to include fields with None values

    Returns:
        List of Document objects ready for indexing
    """
    transformer = PydanticFieldDocumentTransformer(include_null=include_null)
    return list(transformer.transform_documents([
        Document(page_content="", metadata={"structured_doc": model_instance})
    ]))


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
    Person:
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

    url = global_config().get_dsn("vector_store.postgres_url", driver="asyncpg")
    print(url)

    rag = PydanticRag(
        model_definition=EXAMPLE_YAML,
        postgres_url=url,
        embeddings_id="qwen3_06b_deepinfra",
        llm_id=None,
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

"""Data extraction and storage components for the RAG system."""

from typing import Any, Type

from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from loguru import logger  # noqa: F401
from markpickle import dumps
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.llm_factory import get_llm
from src.ai_core.prompts import def_prompt
from src.ai_core.vector_store_factory import VectorStoreFactory
from src.utils.config_mngr import global_config
from src.utils.pydantic.dyn_model_factory import PydanticModelFactory
from src.utils.pydantic.kv_store import load_object_from_kvstore, save_object_to_kvstore
from src.utils.singleton import once

# Markdown separators for text splitting
# fmt:off
MARKDOWN_SEPARATOR = [                                                                                              
    "\n\n", "\n#", "\n##", "\n###", "\n####", "\n#####", "\n######",                                                
    "\n---", "\n***", "\n|", "\n- ", "\n* ", "\n1. ""\n2. ","\n3. ", "\n```", "\n", " ", ""                                        
]  
# fmt:on


class PydanticRagBase(BaseModel):
    """Base class for RAG system handling document extraction and storage."""

    model_definition: dict
    vector_store_factory: VectorStoreFactory
    llm_id: str | None
    kv_store_id: str | None = None

    _top_class: Type[BaseModel] = PrivateAttr()
    _key_field: str = PrivateAttr()
    _llm: Any = PrivateAttr()
    _vector_store: VectorStore = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    @once
    def get_vector_store_factory() -> VectorStoreFactory:
        """Initialize the vector store with embeddings model."""
        EMBEDDINGS_ID = "qwen3_06b_deepinfra"
        postgres_url = global_config().get_dsn("vector_store.postgres_url", driver="asyncpg")
        vector_store_factory = VectorStoreFactory(
            id="PgVector",
            embeddings_factory=EmbeddingsFactory(embeddings_id=EMBEDDINGS_ID),
            config={
                "postgres_url": postgres_url,
                "hybrid_search": True,
                "metadata_columns": [
                    {"name": "entity_id", "data_type": "VARCHAR"},
                    {"name": "field_name", "data_type": "VARCHAR"},
                    {"name": "model_class", "data_type": "VARCHAR"},
                ],
            },
            table_name_prefix="pydantic_fields",
        )
        return vector_store_factory

    def model_post_init(self, __context: Any) -> None:
        """Initialize with configuration and create required components."""
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
        self._vector_store = self.vector_store_factory.get()

    async def abatch_analyze_documents(self, document_ids: list[str], markdown_contents: list[str]) -> list[BaseModel]:
        """Analyze multiple markdown documents asynchronously in batch."""
        from langchain_core.runnables import RunnableConfig

        # Check cache first
        analyzed_docs: list[BaseModel] = []
        remaining_ids: list[str] = []
        remaining_contents: list[str] = []

        if self.kv_store_id:
            for doc_id, content in zip(document_ids, markdown_contents, strict=True):
                cached_doc = load_object_from_kvstore(self.get_top_class(), key=doc_id, kv_store_id=self.kv_store_id)
                if cached_doc:
                    analyzed_docs.append(cached_doc)
                else:
                    remaining_ids.append(doc_id)
                    remaining_contents.append(content)
        else:
            remaining_ids = document_ids
            remaining_contents = markdown_contents

        if not remaining_ids:
            return analyzed_docs

        # Prepare batch processing
        system = f"""
            Extract structured information from the document below into {self._top_class.__name__} schema. 
            Answer with a JSON document. Avoid explanations or extra text."""
        user = "Document to analyze:\n---\n{input}\n---"
        chain = def_prompt(system=system, user=user) | self._llm.with_structured_output(self._top_class)

        try:
            # Process batch asynchronously
            batch_results = await chain.abatch(
                [{"input": content} for content in remaining_contents], config=RunnableConfig(max_concurrency=5)
            )

            # Store results and save to cache
            for doc_id, result in zip(remaining_ids, batch_results, strict=False):
                result.__setattr__("document_id", doc_id)
                analyzed_docs.append(result)
                if self.kv_store_id:
                    save_object_to_kvstore(doc_id, result, kv_store_id="file")

        except Exception as batch_error:
            logger.error(f"Batch analysis failed: {batch_error}, falling back to individual processing")
            # Fallback to individual processing
            for doc_id, content in zip(remaining_ids, remaining_contents, strict=False):
                try:
                    result = await chain.ainvoke({"input": content})
                    result.__setattr__("document_id", doc_id)
                    analyzed_docs.append(result)
                    if self.kv_store_id:
                        save_object_to_kvstore(doc_id, result, kv_store_id="file")
                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {e}")

        return analyzed_docs

    def analyze_document(self, document_id: str, markdown: str) -> BaseModel:
        """Analyze markdown document and return structured data (synchronous wrapper)."""
        from langchain_core.runnables.config import run_in_executor

        # Create a sync wrapper that runs the async method
        def sync_wrapper():
            return asyncio.run(self.abatch_analyze_documents([document_id], [markdown]))

        # Run in executor and get results
        results = run_in_executor(None, sync_wrapper)
        return results[0] if results else None

    def get_top_class(self) -> type[BaseModel]:
        return self._top_class

    def store(self, model_instance: BaseModel) -> None:
        """Process and store a model instance's data in the vector store."""
        chunks = self.chunck(model_instance)
        self.store_chunks(chunks)

    def store_chunks(self, chunks: list[Document]) -> None:
        """Store a document's field embeddings in vector store."""
        self._vector_store.add_documents(chunks)

    def chunck(self, model_instance: BaseModel) -> list[Document]:
        """Generate LangChain Document objects for each field in a Pydantic model."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        documents = []
        model_data = model_instance.model_dump()
        document_id = getattr(model_instance, "document_id", None)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            length_function=len,
            separators=MARKDOWN_SEPARATOR,
            is_separator_regex=False,
        )

        for field_name, field_value in model_data.items():
            if field_value is None:
                continue
            field_info = type(model_instance).model_fields.get(field_name)
            if field_info is None:
                continue

            serialized_content = f"{dumps(field_value)}"
            chunks = text_splitter.split_text(serialized_content)

            for chunk in chunks:
                field_doc = Document(
                    page_content=chunk,
                    metadata={
                        "field_name": field_name,
                        "model_class": model_instance.__class__.__name__,
                        "description": getattr(field_info, "description", "") or "",
                        "document_id": document_id,
                        "entity_id": self.get_key(model_instance),
                    },
                )
                documents.append(field_doc)
        return documents

    def get_key(self, obj: BaseModel) -> str:
        """Extract the actual key value from a model instance."""
        key_path = self._key_field.split(".")
        current_value = obj
        for key_part in key_path:
            if isinstance(current_value, dict):
                current_value = current_value.get(key_part)
            else:
                current_value = getattr(current_value, key_part, None)
            if current_value is None and key_part == key_path[0]:
                current_value = obj
        if current_value is None:
            raise ValueError(f"incorrect key {self._key_field}")
        return str(current_value)

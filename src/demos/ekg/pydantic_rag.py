from typing import Any, List, Optional, Type, TypeVar

from devtools import debug  # noqa: F401
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from loguru import logger
from markpickle import dumps
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.ai_core.embeddings_factory import EmbeddingsFactory
from src.ai_core.llm_factory import get_llm
from src.ai_core.prompts import dedent_ws, def_prompt
from src.ai_core.vector_store_factory import VectorStoreFactory
from src.utils.config_mngr import global_config
from src.utils.pydantic.dyn_model_factory import PydanticModelFactory
from src.utils.pydantic.kv_store import load_object_from_kvstore, save_object_to_kvstore
from src.utils.singleton import once

T = TypeVar("T", bound=BaseModel)


class PydanticRag(BaseModel):
    """RAG system for analyzing and querying structured documents using Pydantic models and embeddings.

    Attributes:
        model_definition: YAML string defining the Pydantic model
        postgres_url: PostgreSQL connection URL
        embeddings_id: ID for embeddings model
        llm_id: ID for LLM used in analysis
    """

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
        # KV_STORE = None
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
        self._vector_store = self.vector_store_factory.get()

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

    def get_top_class_description(self) -> str:
        """Return the description field associated with the top class in the schema."""
        return self.model_definition.get("schema", {}).get(self._top_class.__name__, {}).get("description", "")

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

    def get_key_description(self) -> str:
        """Return the description of the key field from the model definition."""
        key_path = self._key_field.split(".")
        current_schema = self.model_definition.get("schema", {})

        # Start from the top class (RainbowProjectAnalysis)
        top_class_name = self.model_definition.get("top_class", "")
        if not top_class_name or top_class_name not in current_schema:
            return ""

        # Navigate through the schema to get the description
        description = ""
        temp_schema = current_schema

        # First, get the top class fields
        if "fields" in current_schema[top_class_name]:
            temp_schema = current_schema[top_class_name]["fields"]

        for i, key_part in enumerate(key_path):
            if key_part in temp_schema:
                field_def = temp_schema[key_part]

                # Check if this is the last part of the path
                if i == len(key_path) - 1:
                    description = field_def.get("description", "")
                    break

                # Check if it's a reference to another class
                if "type" in field_def:
                    field_type = field_def["type"]
                    # Remove array brackets if present
                    if field_type.startswith("list[") and field_type.endswith("]"):
                        field_type = field_type[5:-1]

                    if field_type in current_schema:
                        # Move to the referenced class's fields
                        if "fields" in current_schema[field_type]:
                            temp_schema = current_schema[field_type]["fields"]
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                # Key part not found
                break

        return description

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
                    "entity_id": self.get_key(model_instance),
                },
            )
            documents.append(field_doc)
        return documents

    def create_vector_search_tool(self) -> BaseTool:
        """Create a LangChain BaseTool for searching the vector store.

        Returns:
            A BaseTool that can search the vector store for semantic matches.
        """

        _entity_id_name: str = self._key_field.split(".")[-1]
        _top_class_description: dict = self.get_top_class_fields()
        _entity_key_name: str = self._key_field.split(".")[-1]

        class VectorSearchInput(BaseModel):
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

        class VectorSearchTool(BaseTool):
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
            args_schema: Optional[ArgsSchema] = VectorSearchInput

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

                if entity_filter:
                    filter_dict = {"$and": [section_filter, entity_filter]}
                else:
                    filter_dict = section_filter

                debug(filter_dict)
                filter_dict = {}

                docs = self._vector_store.similarity_search(query, k=4)
                if not docs:
                    return "No information found"
                entity_id = docs[0].metadata.get("entity_id", "unknown")

                # Group documents by field_name
                fields_dict = {}
                for doc in docs:
                    field_name = doc.metadata.get("field_name", "")
                    if field_name not in fields_dict:
                        fields_dict[field_name] = []
                    fields_dict[field_name].append(doc)

                result_parts = [f"# {_entity_key_name}: {entity_id}"]
                
                # Sort field names according to their order in get_top_class_fields()
                field_order = list(self.get_top_class_fields().keys())
                sorted_fields = sorted(fields_dict.keys(), key=lambda x: field_order.index(x) if x in field_order else len(field_order))
                
                for field_name in sorted_fields:
                    field_docs = fields_dict[field_name]
                    result_parts.append(f"## {field_name}")
                    for doc in field_docs:
                        content = doc.page_content.replace("#", "###")
                        result_parts.append(f"\n{content}")

                return "\n".join(result_parts)

        return VectorSearchTool()

    def get_top_class_fields(self) -> dict:
        return {
            field_name: getattr(field_info, "description", "")
            for field_name, field_info in self._top_class.model_fields.items()
            if field_name != "source"
        }

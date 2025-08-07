# Example usage:
# project = ProjectAnalysis.parse_raw(llm_json_output)
from datetime import date
from typing import Dict, List

import yaml
from langchain.schema import Document
from pydantic import BaseModel

from src.ai_core.embeddings_factory import get_embeddings


def generate_field_embeddings(
    model_instance: BaseModel,
    embeddings_id: str,
    include_null: bool = False,
) -> Dict[str, list[float]]:
    """Generate embeddings for each field in a Pydantic model instance.

    Uses generate_field_documents to create documents for each field, then
    generates embeddings from the document content.

    Args:
        model_instance: An instance of a Pydantic model
        embeddings: LangChain embeddings instance to use for generating vectors
        include_null: Whether to include fields with None values in the output

    Returns:
        Dictionary mapping field names to their embedding vectors
    """
    documents = generate_field_documents(model_instance, include_null)
    embeddings_dict = {}

    embeddings = get_embeddings(embeddings_id)

    for doc in documents:
        field_name = doc.metadata["field_name"]
        embedding = embeddings.embed_query(doc.page_content)
        embeddings_dict[field_name] = embedding

    return embeddings_dict


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
    documents = []

    for field_name, field_value in model_instance.model_dump().items():
        if field_value is None and not include_null:
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

        doc = Document(
            page_content=yaml_content,
            metadata={
                "field_name": field_name,
                "model_class": model_instance.__class__.__name__,
                "description": field_info.description or "",
                "type": str(type(field_value).__name__) if field_value is not None else "None",
            },
        )
        documents.append(doc)

    return documents


def main() -> None:
    from devtools import debug
    from langchain_community.embeddings import FakeEmbeddings
    from rich import print

    from src.utils.config_mngr import global_config

    demo = (
        global_config()
        .merge_with("config/demos/document_extractor.yaml")
        .get_dict("Document_extractor_demo.1", expected_keys=["schema", "key", "top_class"])
    )
    m = create_class_from_dict(demo["schema"], demo["top_class"])

    """Quick test of the embedding utilities using fake embeddings."""
    # Create fake embeddings for testing
    fake_embeddings = FakeEmbeddings(size=1536)


    # Test field embeddings
    print("Generating field embeddings...")
    field_embeddings = generate_field_embeddings(sample_project, "embeddings_768_fake")
    print(f"Generated embeddings for {len(field_embeddings)} fields")

    # Test field documents

    print("\nGenerating field documents...")
    documents = generate_field_documents(sample_project)
    print(f"Generated {len(documents)} documents")
    for doc in documents[:3]:  # Show first 3
        # print(f"- {doc.metadata['field_name']}: {doc.page_content[:100]}...")
        debug(doc)

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()

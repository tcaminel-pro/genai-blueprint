# Example usage:
# project = ProjectAnalysis.parse_raw(llm_json_output)
from datetime import date
from typing import Dict, List

import yaml
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.embeddings import FakeEmbeddings
from pydantic import BaseModel

import src.demos.ekg.rainbow_model as m


def generate_field_embeddings(
    model_instance: BaseModel,
    embeddings: Embeddings,
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
    from rich import print
    from devtools import debug

    """Quick test of the embedding utilities using fake embeddings."""
    # Create fake embeddings for testing
    fake_embeddings = FakeEmbeddings(size=1536)

    # Create a sample project
    sample_project = m.RainbowProjectAnalysis(
        identification=m.ProjectIdentification(
            name="AI Agent Development Project",
            customer="TechCorp Inc",
            status="Pursuit",
            start_date=date(2024, 8, 1),
            end_date=date(2024, 12, 31),
        ),
        description=m.ProjectDescription(
            objectives=["Build AI agents for customer service", "Reduce response time by 50%"],
            scope="End-to-end AI agent implementation",
            success_metrics=["Response time < 2min", "95% accuracy rate"],
        ),
        team=[m.PersonRole(name="John Doe", role="Project Manager", organization="Our Company")],
        delivery=m.DeliveryInfo(
            business_lines=["AI Solutions", "Consulting"],
            locations=["New York", "London"],
            technologies=["Python", "LangChain", "OpenAI"],
        ),
        financials=m.FinancialMetrics(tcv=500000.0, annual_revenue=500000.0, project_margin=25.0),
        risks=[m.RiskAnalysis(risk_description="Tight timeline", mitigation_strategy="Add resources")],
        competition=m.CompetitiveLandscape(competitors=["BigConsulting Inc"], competitive_position="Strong"),
        bidding=m.BiddingStrategy(strategy_type="Prime", win_themes=["Technical expertise", "Experience"]),
        similarity=m.SimilarityAttributes(keywords=["AI", "automation", "customer service"]),
        source="Test data",
    )

    # Test field embeddings
    print("Generating field embeddings...")
    field_embeddings = generate_field_embeddings(sample_project, fake_embeddings)
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

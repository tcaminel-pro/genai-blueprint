from datetime import date
from typing import Dict, List

import yaml
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.embeddings import FakeEmbeddings
from pydantic import BaseModel, Field


class ProjectIdentification(BaseModel):
    """Core project identification information"""

    name: str = Field(description="Official name of the project/opportunity")
    opportunity_id: str | None = Field(None, description="Unique identifier from Salesforce or similar CRM")
    customer: str = Field(description="Client organization name")
    customer_segment: str | None = Field(None, description="Industry vertical of the customer")
    status: str | None = Field(None, description="Current phase: Pursuit/Active/Completed/Cancelled")
    start_date: date | None = Field(None, description="Planned/projected start date")
    end_date: date | None = Field(None, description="Planned/projected end date")


class ProjectDescription(BaseModel):
    """Detailed description of project characteristics"""

    objectives: List[str] | None = Field(None, description="Key business objectives of the project")
    scope: str | None = Field(None, description="Scope of work description")
    success_metrics: List[str] | None = Field(None, description="Measurable success criteria")
    differentiators: List[str] | None = Field(None, description="Unique selling points vs competition")


class PersonRole(BaseModel):
    """Representation of involved parties and their roles"""

    name: str | None = Field(None, description="Full name of involved person")
    role: str | None = Field(None, description="Formal role in project")
    organization: str | None = Field(None, description="Affiliated organization")
    contact_type: str | None = Field("Internal", description="Internal/External/Client/Partner")


class DeliveryInfo(BaseModel):
    """Operational delivery information"""

    business_lines: List[str] | None = Field(None, description="Involved business units")
    locations: List[str] | None = Field(None, description="Geographic locations of delivery")
    partners: List[str] | None = Field(None, description="Third-party partners/subcontractors")
    technologies: List[str] | None = Field(None, description="Key technologies/platforms used")


class FinancialMetrics(BaseModel):
    """Quantitative financial aspects"""

    tcv: float | None = Field(None, description="Total Contract Value in EUR")
    annual_revenue: float | None = Field(None, description="Yearly revenue projection")
    project_margin: float | None = Field(None, description="Gross margin percentage")
    payment_terms: str | None = Field(None, description="Payment schedule and conditions")


class RiskAnalysis(BaseModel):
    """Risk and mitigation information"""

    risk_description: str | None = Field(None, description="Nature of the risk")
    impact_level: str | None = Field("Medium", description="High/Medium/Low impact")
    mitigation_strategy: str | None = Field(None, description="Planned mitigation approach")
    status: str | None = Field("Open", description="Current status of risk mitigation")


class CompetitiveLandscape(BaseModel):
    """Competitive environment analysis"""

    competitors: List[str] | None = Field(None, description="Identified competitors")
    competitive_position: str | None = Field(None, description="Position vs competition")
    key_differentiators: List[str] | None = Field(None, description="Competitive advantages")


class BiddingStrategy(BaseModel):
    """Bidding process details"""

    strategy_type: str | None = Field(None, description="Prime/Sub/JV bidding strategy")
    win_themes: List[str] | None = Field(None, description="Key winning strategy elements")
    pricing_strategy: str | None = Field(None, description="Cost-led/Value-based pricing")
    challenges: List[str] | None = Field(None, description="Key bidding challenges")


class SimilarityAttributes(BaseModel):
    """Attributes for project similarity search"""

    keywords: List[str] | None = Field(None, description="Key terms for semantic search")
    tech_stack_fingerprint: List[str] | None = Field(None, description="Technology combinations")


class RainbowProjectAnalysis(BaseModel):
    """Complete project analysis structure"""

    identification: ProjectIdentification
    description: ProjectDescription
    team: List[PersonRole] = Field(description="All involved personnel")
    delivery: DeliveryInfo
    financials: FinancialMetrics
    risks: List[RiskAnalysis] = Field(description="")
    competition: CompetitiveLandscape
    bidding: BiddingStrategy
    similarity: SimilarityAttributes
    source: str = Field(description="Source document metadata")


# Example usage:
# project = ProjectAnalysis.parse_raw(llm_json_output)


def generate_field_embeddings(
    model_instance: BaseModel,
    embeddings: Embeddings,
    include_null: bool = False,
) -> Dict[str, list[float]]:
    """Generate embeddings for each field in a Pydantic model instance.

    Serializes each field to YAML and creates embeddings using the provided
    LangChain embeddings model.

    Args:
        model_instance: An instance of a Pydantic model
        embeddings: LangChain embeddings instance to use for generating vectors
        include_null: Whether to include fields with None values in the output

    Returns:
        Dictionary mapping field names to their embedding vectors
    """
    embeddings_dict = {}

    for field_name, field_value in model_instance.model_dump().items():
        if field_value is None and not include_null:
            continue

        # Get the field info to include in YAML serialization
        field_info = type(model_instance).model_fields[field_name]

        # Create a YAML representation
        yaml_content = yaml.dump(
            {
                field_name: {
                    "value": field_value,
                    "description": field_info.description,
                    "type": str(type(field_value).__name__) if field_value is not None else "None",
                }
            },
            default_flow_style=False,
            sort_keys=False,
        )

        # Generate embedding
        embedding = embeddings.embed_query(yaml_content)
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
    """Quick test of the embedding utilities using fake embeddings."""
    # Create fake embeddings for testing
    fake_embeddings = FakeEmbeddings(size=1536)

    # Create a sample project
    sample_project = RainbowProjectAnalysis(
        identification=ProjectIdentification(
            name="AI Agent Development Project",
            customer="TechCorp Inc",
            status="Pursuit",
            start_date=date(2024, 8, 1),
            end_date=date(2024, 12, 31),
        ),
        description=ProjectDescription(
            objectives=["Build AI agents for customer service", "Reduce response time by 50%"],
            scope="End-to-end AI agent implementation",
            success_metrics=["Response time < 2min", "95% accuracy rate"],
        ),
        team=[PersonRole(name="John Doe", role="Project Manager", organization="Our Company")],
        delivery=DeliveryInfo(
            business_lines=["AI Solutions", "Consulting"],
            locations=["New York", "London"],
            technologies=["Python", "LangChain", "OpenAI"],
        ),
        financials=FinancialMetrics(tcv=500000.0, annual_revenue=500000.0, project_margin=25.0),
        risks=[RiskAnalysis(risk_description="Tight timeline", mitigation_strategy="Add resources")],
        competition=CompetitiveLandscape(competitors=["BigConsulting Inc"], competitive_position="Strong"),
        bidding=BiddingStrategy(strategy_type="Prime", win_themes=["Technical expertise", "Experience"]),
        similarity=SimilarityAttributes(keywords=["AI", "automation", "customer service"]),
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
        print(f"- {doc.metadata['field_name']}: {doc.page_content[:100]}...")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()

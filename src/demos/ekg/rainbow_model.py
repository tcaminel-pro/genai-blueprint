from datetime import date
from typing import List

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

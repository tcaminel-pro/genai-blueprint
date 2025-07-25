from datetime import date
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Make all fields optional  (with .. | None ...  = None) AI!
class ProjectIdentification(BaseModel):
    """Core project identification information"""

    name: str = Field(..., description="Official name of the project/opportunity")
    opportunity_id: str = Field(..., description="Unique identifier from Salesforce or similar CRM")
    customer: str = Field(..., description="Client organization name")
    customer_segment: Optional[str] = Field(None, description="Industry vertical of the customer")
    status: str = Field(..., description="Current phase: Pursuit/Active/Completed/Cancelled")
    start_date: Optional[date] = Field(None, description="Planned/projected start date")
    end_date: Optional[date] = Field(None, description="Planned/projected end date")


class ProjectDescription(BaseModel):
    """Detailed description of project characteristics"""

    objectives: List[str] = Field(..., description="Key business objectives of the project")
    scope: str = Field(..., description="Scope of work description")
    success_metrics: List[str] = Field(..., description="Measurable success criteria")
    differentiators: List[str] = Field(..., description="Unique selling points vs competition")


class PersonRole(BaseModel):
    """Representation of involved parties and their roles"""

    name: str = Field(..., description="Full name of involved person")
    role: str = Field(..., description="Formal role in project")
    organization: Optional[str] = Field(None, description="Affiliated organization")
    contact_type: str = Field("Internal", description="Internal/External/Client/Partner")


class DeliveryInfo(BaseModel):
    """Operational delivery information"""

    business_lines: List[str] = Field(..., description="Involved business units")
    locations: List[str] = Field(..., description="Geographic locations of delivery")
    partners: List[str] = Field(..., description="Third-party partners/subcontractors")
    technologies: List[str] = Field(..., description="Key technologies/platforms used")


class FinancialMetrics(BaseModel):
    """Quantitative financial aspects"""

    tcv: Optional[float] = Field(None, description="Total Contract Value in EUR")
    annual_revenue: Optional[float] = Field(None, description="Yearly revenue projection")
    project_margin: Optional[float] = Field(None, description="Gross margin percentage")
    payment_terms: Optional[str] = Field(None, description="Payment schedule and conditions")


class RiskAnalysis(BaseModel):
    """Risk and mitigation information"""

    risk_description: str = Field(..., description="Nature of the risk")
    impact_level: str = Field("Medium", description="High/Medium/Low impact")
    mitigation_strategy: Optional[str] = Field(None, description="Planned mitigation approach")
    status: str = Field("Open", description="Current status of risk mitigation")


class CompetitiveLandscape(BaseModel):
    """Competitive environment analysis"""

    competitors: List[str] = Field(..., description="Identified competitors")
    competitive_position: str = Field(..., description="Position vs competition")
    key_differentiators: List[str] = Field(..., description="Competitive advantages")


class BiddingStrategy(BaseModel):
    """Bidding process details"""

    strategy_type: str = Field(..., description="Prime/Sub/JV bidding strategy")
    win_themes: List[str] = Field(..., description="Key winning strategy elements")
    pricing_strategy: str = Field(..., description="Cost-led/Value-based pricing")
    challenges: List[str] = Field(..., description="Key bidding challenges")


class SimilarityAttributes(BaseModel):
    """Attributes for project similarity search"""

    keywords: List[str] = Field(..., description="Key terms for semantic search")
    tech_stack_fingerprint: List[str] = Field(..., description="Technology combinations")
    embedding_vector: Optional[List[float]] = Field(None, description="Vector embedding for ANN search")


class RainbowProjectAnalysis(BaseModel):
    """Complete project analysis structure"""

    identification: ProjectIdentification
    description: ProjectDescription
    team: List[PersonRole] = Field(..., description="All involved personnel")
    delivery: DeliveryInfo
    financials: FinancialMetrics
    risks: List[RiskAnalysis]
    competition: CompetitiveLandscape
    bidding: BiddingStrategy
    similarity: SimilarityAttributes
    source: Dict[str, Union[str, date]] = Field(..., description="Source document metadata")


# Example usage:
# project = ProjectAnalysis.parse_raw(llm_json_output)

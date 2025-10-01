"""
CV-related DataPoint models for Cognee low-level API.
These classes define the structure for CV/Resume data that integrates with Cognee's knowledge graph.
"""

from typing import List, Optional

from cognee.low_level import DataPoint


class ContactInfo(DataPoint):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    metadata: dict = {"index_fields": ["email"]}


class Education(DataPoint):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None
    metadata: dict = {"index_fields": ["institution", "degree", "field_of_study"]}


class WorkExperience(DataPoint):
    company: str
    position: str
    start_date: Optional[str] = None  # Format: YYYY-MM or YYYY
    end_date: Optional[str] = None  # Format: YYYY-MM or YYYY, or "Present"
    location: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    achievements: Optional[List[str]] = None
    metadata: dict = {"index_fields": ["company", "position"]}


class Skill(DataPoint):
    name: str
    category: Optional[str] = None  # e.g., "Programming", "Frameworks", "Tools", "Languages"
    proficiency_level: Optional[str] = None  # e.g., "Beginner", "Intermediate", "Advanced", "Expert"
    metadata: dict = {"index_fields": ["name", "category"]}


class Project(DataPoint):
    name: str
    description: str
    technologies: Optional[List[str]] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "technologies"]}


class Certification(DataPoint):
    name: str
    issuer: str
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "issuer"]}


class Company(DataPoint):
    name: str
    industry: Optional[str] = None
    location: Optional[str] = None
    size: Optional[str] = None  # e.g., "Small", "Medium", "Large", "Enterprise"
    metadata: dict = {"index_fields": ["name", "industry"]}


class Institution(DataPoint):
    name: str
    type: str = "Educational Institution"  # University, College, etc.
    location: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "type"]}


class CVPerson(DataPoint):
    name: str
    contact_info: Optional[ContactInfo] = None
    summary: Optional[str] = None

    # Relationships
    education: Optional[List[Education]] = None
    work_experience: Optional[List[WorkExperience]] = None
    skills: Optional[List[Skill]] = None
    projects: Optional[List[Project]] = None
    certifications: Optional[List[Certification]] = None

    # Simple fields
    languages: Optional[List[str]] = None
    interests: Optional[List[str]] = None

    metadata: dict = {"index_fields": ["name", "summary"]}


class SkillCategory(DataPoint):
    name: str  # e.g., "Programming Languages", "Frameworks", "Tools"
    description: Optional[str] = None
    metadata: dict = {"index_fields": ["name"]}


class Industry(DataPoint):
    name: str
    description: Optional[str] = None
    metadata: dict = {"index_fields": ["name"]}

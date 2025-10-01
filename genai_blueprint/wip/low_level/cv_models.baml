// CV-related BAML models for structured data extraction
// These models define the structure for CV/Resume data that can be extracted from text

class ContactInfo {
    email string?
    phone string?
    linkedin string?
    github string?
    website string?
}

class Education {
    institution string
    degree string
    field_of_study string?
    graduation_year int?
    gpa float?
}

class WorkExperience {
    company string
    position string
    start_date string?  // Format: YYYY-MM or YYYY
    end_date string?    // Format: YYYY-MM or YYYY, or "Present"
    location string?
    responsibilities string[]?
    achievements string[]?
}

class Skill {
    name string
    category string?    // e.g., "Programming", "Frameworks", "Tools", "Languages"
    proficiency_level string?  // e.g., "Beginner", "Intermediate", "Advanced", "Expert"
}

class Project {
    name string
    description string
    technologies string[]?
    role string?
    duration string?
}

class Certification {
    name string
    issuer string
    issue_date string?
    expiry_date string?
    credential_id string?
}

class CVProfile {
    name string
    contact_info ContactInfo?
    summary string?
    education Education[]?
    work_experience WorkExperience[]?
    skills Skill[]?
    projects Project[]?
    certifications Certification[]?
    languages string[]?
    interests string[]?
}

// Function to extract CV data from unstructured text
function ExtractCVData(content: string) -> CVProfile {
    client GPT4Turbo
    prompt #"
        You are an expert CV/Resume parser. Extract structured information from the provided CV/Resume text.
        
        Instructions:
        1. Extract all relevant information and organize it into the CVProfile structure
        2. For dates, use YYYY-MM format when possible, or just YYYY if month is not available
        3. Use "Present" for current positions
        4. Categorize skills appropriately (Programming, Frameworks, Tools, Languages, etc.)
        5. Extract achievements and responsibilities separately when possible
        6. If information is not available, leave the field empty rather than guessing
        
        CV Text:
        {{ content }}
    "#
}

// Function to extract work experience relationships for knowledge graph
function ExtractCVRelationships(cv_content: string) -> KnowledgeGraph {
    client GPT4Turbo
    prompt #"
        Extract relationships from CV data to build a knowledge graph.
        Focus on:
        - Person -> worked_at -> Company relationships
        - Person -> has_skill -> Skill relationships  
        - Person -> studied_at -> Institution relationships
        - Person -> worked_on -> Project relationships
        - Company -> operates_in -> Industry relationships
        - Skill -> belongs_to_category -> SkillCategory relationships
        
        Use the CV content below:
        {{ cv_content }}
    "#
}
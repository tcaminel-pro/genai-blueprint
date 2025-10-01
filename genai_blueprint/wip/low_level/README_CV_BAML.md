# CV Processing with BAML and Cognee Low-Level API

This directory contains examples demonstrating how to use BAML (Behavioral Abstraction Markup Language) framework with Cognee's low-level API to process CV/Resume data instead of using pre-structured JSON.

## Overview

The examples show how to:
1. Define BAML models for CV data structures
2. Use BAML for intelligent extraction from unstructured CV text
3. Create DataPoint classes that integrate with Cognee's knowledge graph
4. Build rich relationships between CV entities
5. Perform advanced searches on CV data

## Key Differences from JSON Approach

| Aspect | JSON Approach | BAML Approach |
|--------|---------------|---------------|
| Data Input | Pre-structured JSON files | Unstructured CV text |
| Extraction | Manual JSON parsing | LLM-powered intelligent extraction |
| Flexibility | Fixed schema | Adaptive schema with dynamic fields |
| Accuracy | Depends on manual structuring | LLM understands context and nuance |
| Scalability | Requires manual data preparation | Automated extraction from any CV format |

## Files Description

### Core Files

1. **`cv_models.baml`** - BAML model definitions for CV data structures
   - Defines ContactInfo, Education, WorkExperience, Skill, Project, etc.
   - Includes extraction functions for CV parsing
   - Supports both structured data extraction and knowledge graph relationships

2. **`cv_models.py`** - Python DataPoint classes corresponding to BAML models
   - CVPerson, Education, WorkExperience, Skill, Project, etc.
   - Includes metadata for vector search indexing
   - Defines relationships between entities

3. **`cv_data.json`** - Sample structured CV data (for comparison/fallback)

### Example Scripts

4. **`cv_pipeline_baml.py`** - Basic BAML integration example
   - Shows fundamental BAML usage with CV data
   - Demonstrates entity creation and relationship building
   - Includes visualization and basic search capabilities

5. **`cv_pipeline_baml_advanced.py`** - Advanced BAML integration example
   - Full-featured BAML CV extraction
   - Rich knowledge graph creation
   - Advanced search demonstrations
   - Industry and company size inference

## BAML Model Structure

The BAML models define a comprehensive CV data structure:

```
CVProfile
├── ContactInfo (email, phone, linkedin, github)
├── Education[] (institution, degree, field, year)
├── WorkExperience[] (company, position, dates, responsibilities)
├── Skills[] (name, category, proficiency_level)
├── Projects[] (name, description, technologies, role)
├── Certifications[] (name, issuer, dates)
├── Languages[]
└── Interests[]
```

## Knowledge Graph Relationships

The system creates rich relationships between entities:

- **Person** → `worked_at` → **Company**
- **Person** → `has_skill` → **Skill**
- **Person** → `studied_at` → **Institution**
- **Person** → `worked_on` → **Project**
- **Company** → `operates_in` → **Industry**
- **Skill** → `belongs_to_category` → **SkillCategory**
- **Project** → `uses_technology` → **Technology**

## Usage Instructions

### Prerequisites

1. Ensure Cognee is properly installed and configured
2. Set up your LLM provider (OpenAI, Anthropic, etc.) in Cognee config
3. Configure BAML framework in your Cognee setup

### Running the Examples

#### Basic Example
```bash
cd examples/low_level
python cv_pipeline_baml.py
```

#### Advanced Example
```bash
cd examples/low_level
python cv_pipeline_baml_advanced.py
```

### Setting Up BAML Framework

1. **Configure LLM Provider**:
   ```python
   import cognee
   cognee.config.set_llm_config({
       "structured_output_framework": "BAML",
       "llm_provider": "openai",
       "llm_model": "gpt-4",
       "llm_api_key": "your-api-key"
   })
   ```

2. **Add BAML Models**: 
   - Place your `.baml` files in the BAML source directory
   - Run BAML compilation to generate Python clients

3. **Use BAML Extraction**:
   ```python
   from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b
   
   # Extract CV data
   cv_data = await b.ExtractCVData(cv_text_content)
   ```

## Example Queries

After running the pipeline, you can query the knowledge graph:

1. **Skill-based queries**:
   - "Who has Python expertise?"
   - "Find candidates with machine learning experience"
   - "Who knows TensorFlow?"

2. **Experience-based queries**:
   - "Who worked at technology companies?"
   - "Find senior data scientists"
   - "Who has startup experience?"

3. **Education-based queries**:
   - "Who graduated from Stanford?"
   - "Find candidates with Computer Science degrees"
   - "Who has a Ph.D.?"

4. **Project-based queries**:
   - "What projects use deep learning?"
   - "Who built NLP systems?"
   - "Find candidates with cloud experience"

## Benefits of BAML Approach

### 1. **Intelligent Extraction**
- Understands context and nuance in CV text
- Handles various CV formats automatically
- Extracts implicit information (e.g., skill levels from context)

### 2. **Structured Output**
- Type-safe data extraction
- Consistent schema enforcement
- Validation at extraction time

### 3. **Scalability**
- Process any CV format without manual preprocessing
- Batch processing of multiple CVs
- Automatic entity relationship discovery

### 4. **Rich Knowledge Graph**
- Creates meaningful relationships between entities
- Enables complex queries across multiple dimensions
- Supports graph-based recommendations

## Customization

### Adding New CV Fields

1. **Update BAML Model**:
   ```baml
   class CVProfile {
       // ... existing fields
       custom_field string?
   }
   ```

2. **Update Python DataPoint**:
   ```python
   class CVPerson(DataPoint):
       # ... existing fields
       custom_field: Optional[str] = None
   ```

3. **Update Extraction Logic**:
   Modify the extraction prompts to include new fields

### Custom Search Types

Create specialized search functions for your use cases:

```python
async def find_candidates_by_technology(technology: str):
    return await search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=f"Who has experience with {technology}?"
    )
```

## Troubleshooting

### Common Issues

1. **BAML Extraction Fails**:
   - Check LLM provider configuration
   - Verify BAML model compilation
   - Ensure sufficient context in prompts

2. **Knowledge Graph Empty**:
   - Check entity creation logic
   - Verify DataPoint model definitions
   - Ensure pipeline tasks are configured correctly

3. **Search Returns No Results**:
   - Verify data was stored in knowledge graph
   - Check search query format
   - Ensure proper entity relationships

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Custom Relationship Extraction

Use BAML to extract custom relationships:

```baml
function ExtractCVRelationships(cv_content: string) -> KnowledgeGraph {
    client GPT4Turbo
    prompt #"
        Extract relationships from CV data:
        - Person mentored_by Person
        - Person collaborated_with Person
        - Person contributed_to Project
        
        CV: {{ cv_content }}
    "#
}
```

### Multi-Language Support

Extend BAML models for international CVs:

```baml
class CVProfile {
    name string
    languages Language[]
    // ... other fields
}

class Language {
    name string
    proficiency_level string
    native_speaker bool?
}
```

## Contributing

To contribute to these examples:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## Support

For questions or issues:
1. Check the Cognee documentation
2. Review BAML framework documentation
3. Open an issue in the Cognee repository
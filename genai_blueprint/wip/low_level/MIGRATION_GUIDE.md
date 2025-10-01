# Migration Guide: From JSON to BAML Approach

This guide helps you transition from using pre-structured JSON data to BAML-powered intelligent extraction in your Cognee low-level API projects.

## Overview

The migration involves shifting from:
- **JSON Approach**: Pre-structured data → Direct parsing → Knowledge graph
- **BAML Approach**: Unstructured text → LLM extraction → Structured data → Knowledge graph

## Step-by-Step Migration

### 1. Assess Your Current Setup

First, identify what you're currently using:

```python
# Current JSON approach (like original pipeline.py)
class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}

def ingest_files(data: List[Any]):
    # Direct JSON parsing
    for item in data:
        person = Person(name=item["name"])
```

### 2. Define BAML Models

Create BAML model definitions for your domain:

```baml
// cv_models.baml
class Person {
    name string
    summary string?
    skills string[]?
    experience_years int?
}

function ExtractPersonData(content: string) -> Person {
    client GPT4Turbo
    prompt #"
        Extract person information from the text:
        {{ content }}
    "#
}
```

### 3. Create Corresponding DataPoint Classes

Update your Python models to include richer information:

```python
# Enhanced DataPoint class
class Person(DataPoint):
    name: str
    summary: Optional[str] = None
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    metadata: dict = {"index_fields": ["name", "summary", "skills"]}
```

### 4. Replace Data Processing Logic

Transform your ingestion function:

```python
# Before: JSON parsing
def ingest_files(data: List[Any]):
    entities = []
    for item in data:
        person = Person(name=item["name"])
        entities.append(person)
    return entities

# After: BAML extraction
async def ingest_text_with_baml(texts: List[str]):
    entities = []
    for text in texts:
        extracted_data = await extract_with_baml(text)
        person = Person(
            name=extracted_data["name"],
            summary=extracted_data.get("summary"),
            skills=extracted_data.get("skills"),
            experience_years=extracted_data.get("experience_years")
        )
        entities.append(person)
    return entities
```

### 5. Update Pipeline Configuration

Modify your pipeline to handle text inputs:

```python
# Before: JSON data input
data = [{"people": people_json, "companies": companies_json}]

# After: Text data input
texts = [cv_text_1, cv_text_2, cv_text_3]
entities = await ingest_text_with_baml(texts)
```

## Complete Migration Example

Here's a complete before/after comparison:

### Before (JSON Approach)
```python
import json
from cognee.low_level import DataPoint

class Person(DataPoint):
    name: str
    company: str
    metadata: dict = {"index_fields": ["name", "company"]}

def process_json_data():
    # Load JSON file
    with open("people.json") as f:
        data = json.load(f)
    
    # Direct parsing
    entities = []
    for person_data in data:
        person = Person(
            name=person_data["name"],
            company=person_data["company"]
        )
        entities.append(person)
    
    return entities
```

### After (BAML Approach)
```python
from typing import Optional, List
from cognee.low_level import DataPoint
from cognee.infrastructure.llm.LLMGateway import LLMGateway

class Person(DataPoint):
    name: str
    company: Optional[str] = None
    position: Optional[str] = None
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    summary: Optional[str] = None
    metadata: dict = {"index_fields": ["name", "company", "skills", "summary"]}

async def process_cv_text(cv_texts: List[str]):
    entities = []
    
    for cv_text in cv_texts:
        # BAML extraction
        extracted_data = await extract_person_data(cv_text)
        
        person = Person(
            name=extracted_data["name"],
            company=extracted_data.get("company"),
            position=extracted_data.get("position"),
            skills=extracted_data.get("skills"),
            experience_years=extracted_data.get("experience_years"),
            summary=extracted_data.get("summary")
        )
        entities.append(person)
    
    return entities

async def extract_person_data(cv_text: str) -> dict:
    """Extract structured data using BAML framework"""
    # Use BAML client or LLMGateway for extraction
    prompt = f"""
    Extract person information from this CV:
    {cv_text}
    
    Return JSON with: name, company, position, skills[], experience_years, summary
    """
    # Implementation would use actual BAML client
    return await llm_extract(prompt)
```

## Configuration Changes

### 1. LLM Configuration

Set up BAML framework in your configuration:

```python
# config.py
import cognee

cognee.config.set_llm_config({
    "structured_output_framework": "BAML",
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "llm_api_key": "your-api-key"
})
```

### 2. BAML Client Setup

Initialize BAML in your project:

```python
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b

async def setup_baml():
    config = get_llm_config()
    # BAML client is now ready to use
    return config.baml_registry
```

## Data Input Migration

### Input Format Changes

| Aspect | JSON Approach | BAML Approach |
|--------|---------------|---------------|
| **Input Format** | Structured JSON files | Unstructured text (CVs, documents) |
| **Data Preparation** | Manual structuring required | Raw text input |
| **Schema** | Fixed, predefined | Flexible, extracted |
| **Processing** | Parsing only | Intelligent extraction |

### Example Data Transformation

**Before (JSON):**
```json
{
  "name": "John Doe",
  "company": "TechCorp",
  "department": "Engineering"
}
```

**After (Text):**
```text
John Doe
Senior Software Engineer at TechCorp

Experienced developer with 5 years in Python and JavaScript.
Led multiple projects in the Engineering department.
Skills: Python, JavaScript, React, AWS, Docker
```

## Pipeline Updates

### Task Modifications

Update your pipeline tasks:

```python
# Before
pipeline = run_tasks(
    [Task(ingest_files), Task(add_data_points)],
    dataset_id=dataset.id,
    data=[json_data],
    incremental_loading=False,
)

# After
pipeline = run_tasks(
    [Task(ingest_text_with_baml), Task(add_data_points)],
    dataset_id=dataset.id,
    data=cv_texts,
    incremental_loading=False,
)
```

## Error Handling

Add robust error handling for LLM-based extraction:

```python
async def safe_baml_extraction(text: str) -> dict:
    try:
        result = await extract_with_baml(text)
        return result
    except Exception as e:
        print(f"BAML extraction failed: {e}")
        # Fallback to basic text parsing
        return basic_text_parsing(text)

def basic_text_parsing(text: str) -> dict:
    """Fallback parsing for when BAML fails"""
    lines = text.split('\n')
    name = lines[0] if lines else "Unknown"
    return {"name": name}
```

## Testing Your Migration

### 1. Parallel Testing

Run both approaches side by side:

```python
async def test_migration():
    # Test JSON approach
    json_entities = process_json_data()
    
    # Test BAML approach
    baml_entities = await process_cv_text(cv_texts)
    
    # Compare results
    compare_extraction_quality(json_entities, baml_entities)
```

### 2. Validation

Validate extraction quality:

```python
def validate_extraction(extracted_data: dict) -> bool:
    required_fields = ["name"]
    return all(field in extracted_data for field in required_fields)
```

## Performance Considerations

### Cost Management

```python
# Batch processing for efficiency
async def batch_process_cvs(cv_texts: List[str], batch_size: int = 5):
    results = []
    for i in range(0, len(cv_texts), batch_size):
        batch = cv_texts[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            extract_with_baml(text) for text in batch
        ])
        results.extend(batch_results)
    return results
```

### Caching

Implement caching for repeated extractions:

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_baml_extraction(text_hash: str, text: str) -> dict:
    return await extract_with_baml(text)

async def extract_with_cache(text: str) -> dict:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return await cached_baml_extraction(text_hash, text)
```

## Monitoring and Debugging

### Extraction Quality Monitoring

```python
def monitor_extraction_quality(extracted_data: dict, source_text: str):
    """Monitor and log extraction quality metrics"""
    quality_score = calculate_quality_score(extracted_data, source_text)
    
    if quality_score < 0.7:
        print(f"Low quality extraction detected: {quality_score}")
        print(f"Source: {source_text[:100]}...")
        print(f"Extracted: {extracted_data}")
```

### Debug Logging

```python
import logging

logger = logging.getLogger(__name__)

async def debug_baml_extraction(text: str) -> dict:
    logger.info(f"Processing text: {text[:50]}...")
    
    try:
        result = await extract_with_baml(text)
        logger.info(f"Extraction successful: {len(result)} fields")
        return result
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
```

## Gradual Migration Strategy

### Phase 1: Parallel Implementation
- Keep existing JSON approach
- Add BAML processing alongside
- Compare results and quality

### Phase 2: Gradual Transition
- Migrate non-critical data first
- Test with subset of data
- Monitor performance and accuracy

### Phase 3: Full Migration
- Replace JSON processing with BAML
- Remove old code
- Optimize performance

### Phase 4: Enhancement
- Add new extraction capabilities
- Implement advanced relationships
- Extend to new data types

## Common Pitfalls and Solutions

### 1. Over-reliance on LLM
**Problem**: Every small data extraction uses LLM
**Solution**: Use BAML for complex extraction, keep simple parsing for basic fields

### 2. Inconsistent Extraction
**Problem**: LLM returns different formats
**Solution**: Use structured output with schema validation

### 3. High Costs
**Problem**: Too many LLM calls
**Solution**: Implement batching, caching, and selective extraction

### 4. Poor Error Handling
**Problem**: Pipeline fails on extraction errors
**Solution**: Implement fallback parsing and graceful degradation

## Success Metrics

Track these metrics to measure migration success:

- **Data Richness**: Number of fields extracted per entity
- **Accuracy**: Correctness of extracted information
- **Coverage**: Percentage of successful extractions
- **Performance**: Processing time per document
- **Cost**: LLM usage costs
- **Search Quality**: Relevance of search results

## Next Steps

After successful migration:

1. **Expand Extraction**: Add more sophisticated data types
2. **Enhance Relationships**: Create richer entity relationships
3. **Custom Models**: Develop domain-specific BAML models
4. **Advanced Search**: Implement semantic search capabilities
5. **Real-time Processing**: Set up streaming data ingestion

## Support Resources

- **Cognee Documentation**: Core framework documentation
- **BAML Documentation**: BAML framework specifics
- **Examples**: Reference implementations in this directory
- **Community**: Cognee community forums and discussions
# BAML-Based Structured Extraction

This directory contains a BAML-based implementation of the structured extraction functionality that was originally built using LangChain. The new implementation leverages BAML (Boundary Markup Language) for more reliable and type-safe structured output from LLMs.

## Overview

The BAML implementation provides the same functionality as the original `structured_extract` function but with several advantages:

- **Type Safety**: Uses BAML's generated Pydantic models for guaranteed type safety
- **Better Error Handling**: BAML provides more robust error handling for structured output
- **Performance**: Async-first design with proper concurrency control
- **Maintainability**: Schema definitions are centralized in BAML files
- **Testing**: Easier to test and validate with BAML's built-in validation

## Files

- `cli_commands_baml.py` - BAML-based CLI commands (main implementation)
- `test_baml_extract.py` - Test script to validate BAML functionality
- `baml_src/` - BAML schema definitions and configuration
- `baml_client/` - Auto-generated Python client code (do not edit directly)
- `README_BAML.md` - This documentation file

## Schema Definition

The BAML schema is defined in `baml_src/rainbow_project_analysis.baml` and includes:

- **ReviewedOpportunity** - Main project/opportunity data structure
- **Person** - Team member information
- **FinancialMetrics** - Financial data (TCV, margins, etc.)
- **RiskAnalysis** - Project risk assessments
- **DeliveryInfo** - Operational delivery information
- **CompetitiveLandscape** - Competitor analysis
- **BiddingStrategy** - Sales strategy information

## Usage

### CLI Command

The new BAML-based extraction can be used with the `structured_extract_baml` command:

```bash
# Extract from a single file
uv run cli structured-extract-baml document.md

# Extract from a directory recursively
uv run cli structured-extract-baml ./documents/ --recursive

# Force reprocessing of already processed files
uv run cli structured-extract-baml ./documents/ --force

# Process with custom batch size
uv run cli structured-extract-baml ./documents/ --batch-size 10
```

### Python API

You can also use the BAML processor directly in your Python code:

```python
from src.demos.ekg.cli_commands_baml import BamlStructuredProcessor
from pathlib import Path

# Create processor
processor = BamlStructuredProcessor(kvstore_id="file")

# Process a single document
result = processor.analyze_document("doc_id", markdown_content)
print(f"Extracted: {result.name}")

# Process multiple files
md_files = [Path("document1.md"), Path("document2.md")]
await processor.process_files(md_files, batch_size=5)
```

### Direct BAML Client Usage

For more advanced use cases, you can use the BAML client directly:

```python
from src.demos.ekg.baml_client import b as baml_client
from src.demos.ekg.baml_client.async_client import b as baml_async_client

# Synchronous extraction
result = baml_client.ExtractRainbow(rainbow_file=markdown_content)

# Asynchronous extraction
result = await baml_async_client.ExtractRainbow(rainbow_file=markdown_content)
```

## Configuration

### API Keys

Make sure you have the required API key set up:

```bash
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### BAML Configuration

The BAML configuration is defined in:

- `baml_src/clients.baml` - LLM client configurations
- `baml_src/generators.baml` - Code generation settings
- `baml_src/rainbow_project_analysis.baml` - Schema definitions

### Model Configuration

The current configuration uses:
- **Client**: OpenRouter API
- **Model**: `gpt-4.1-mini` (via OpenRouter)
- **Fallback**: `deepseek/deepseek-chat-v3.1`

You can modify the client configuration in `baml_src/clients.baml` to use different models or providers.

## Testing

### Quick Test

To quickly test that BAML is working correctly:

```bash
cd src/demos/ekg
uv run python -m test_baml_extract
```

### Full Integration Test

Test with real markdown files:

```bash
# Create a test markdown file
echo "# Test Project..." > test_document.md

# Run extraction
uv run cli structured-extract-baml test_document.md --force
```

## Comparison with Original Implementation

| Feature | Original (LangChain) | BAML Implementation |
|---------|---------------------|-------------------|
| Schema Definition | Python Pydantic classes | BAML schema files |
| Type Safety | Runtime validation | Compile-time + runtime |
| Error Handling | Basic try/catch | BAML error types |
| Async Support | Yes | Yes (improved) |
| Concurrency Control | Basic batching | Semaphore-controlled |
| Testing | Manual | Built-in validation |
| Model Flexibility | Config-driven | BAML client config |

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure BAML client is regenerated
   ```bash
   cd src/demos/ekg
   uv run baml-cli generate
   ```

2. **API Key Error**: Set the OpenRouter API key
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```

3. **Version Mismatch**: Ensure BAML versions match
   ```bash
   # Check versions in generators.baml and installed baml-py
   uv pip list | grep baml
   ```

4. **Schema Changes**: After modifying BAML files, regenerate the client
   ```bash
   uv run baml-cli generate
   ```

### Debugging

Enable debug logging:

```python
from loguru import logger
logger.add(sys.stderr, level="DEBUG")
```

Or use the test script to validate functionality:

```bash
uv run python -c "
from src.demos.ekg.test_baml_extract import main
import asyncio
asyncio.run(main())
"
```

## Future Improvements

- **Batch Processing**: Optimize for larger document sets
- **Streaming**: Add streaming support for real-time processing
- **Validation**: Add more sophisticated validation rules
- **Caching**: Improve caching strategies for better performance
- **Multi-Model**: Support for multiple LLM providers simultaneously

## Contributing

When making changes:

1. Modify BAML schema in `baml_src/`
2. Regenerate client: `uv run baml-cli generate`
3. Update tests in `test_baml_extract.py`
4. Test with real documents
5. Update this documentation

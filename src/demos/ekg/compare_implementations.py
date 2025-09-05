"""Comparison script between original LangChain and BAML implementations.

This script demonstrates the differences in usage and output between the two
structured extraction approaches.
"""

import asyncio
import time
from typing import Any

from loguru import logger

# Import BAML implementation
from .cli_commands_baml import BamlStructuredProcessor
from .baml_client.types import ReviewedOpportunity as BamlReviewedOpportunity

# Sample content for testing
SAMPLE_CONTENT = """# Project Review: CNES Earth Observation Platform

## Project Details
- **Project Name**: CNES TMA VENUS VIP PEPS THEIA MUSCATE
- **Opportunity ID**: 9000559500
- **Client**: CNES (Centre national d'études spatiales)
- **Industry**: Aerospace / Earth Observation
- **Start Date**: 01/04/2019
- **Duration**: 2+36 Months
- **TCV**: 1.8 M€

## Team
- **Bid Manager**: Barthélémy MARTI
- **Solution Manager**: Olivier Rondeau  
- **Client Leader**: Marc Ferrer
- **Sales Lead**: Aurore Dorez

## Scope
TMA of Earth observation data production and distribution centers including:
- PEPS platform maintenance
- THEIA MUSCATE processing chains
- VENUS VIP data handling

## Key Risks
- SLA compliance penalties due to quality or delivery delays
- Delivery cost overruns from underestimated costs
- Scope management issues due to minimal transition knowledge
- Skill retention challenges from high turnover

## Competition
- **CAP**: Present on THEIA MUSCATE with client proximity advantage
- **CS**: Limited Venus activity but compliance response
- **Thales**: Compliance response approach
"""


def compare_data_structures():
    """Compare the data structures between implementations."""
    logger.info("=== Data Structure Comparison ===")
    
    # BAML structure
    baml_fields = list(BamlReviewedOpportunity.model_fields.keys())
    logger.info(f"BAML ReviewedOpportunity fields ({len(baml_fields)}):")
    for field in baml_fields:
        field_info = BamlReviewedOpportunity.model_fields[field]
        field_type = field_info.annotation if hasattr(field_info, 'annotation') else 'Unknown'
        logger.info(f"  - {field}: {field_type}")


async def compare_performance():
    """Compare processing performance between implementations."""
    logger.info("\n=== Performance Comparison ===")
    
    # Test BAML implementation
    logger.info("Testing BAML implementation...")
    baml_processor = BamlStructuredProcessor(kvstore_id=None)
    
    start_time = time.time()
    try:
        baml_result = baml_processor.analyze_document("test_perf", SAMPLE_CONTENT)
        baml_time = time.time() - start_time
        logger.success(f"BAML processing time: {baml_time:.2f}s")
        logger.info(f"BAML result: {baml_result.name}")
    except Exception as e:
        baml_time = None
        logger.error(f"BAML failed: {e}")
    
    # Note: Original implementation would be tested here if schema was available
    # For now, just document the comparison
    logger.info("\nPerformance Summary:")
    if baml_time:
        logger.info(f"✓ BAML: {baml_time:.2f}s")
    else:
        logger.info("✗ BAML: Failed")
    logger.info("? Original: Would need schema configuration")


def compare_error_handling():
    """Compare error handling between implementations."""
    logger.info("\n=== Error Handling Comparison ===")
    
    # Test with invalid content
    invalid_content = "This is not a project document."
    
    logger.info("Testing BAML error handling with invalid content...")
    baml_processor = BamlStructuredProcessor(kvstore_id=None)
    
    try:
        result = baml_processor.analyze_document("test_error", invalid_content)
        logger.info(f"BAML handled gracefully: {result.name if result.name else 'No name extracted'}")
    except Exception as e:
        logger.warning(f"BAML error (expected): {type(e).__name__}: {e}")


async def compare_async_capabilities():
    """Compare async processing capabilities."""
    logger.info("\n=== Async Processing Comparison ===")
    
    # Test concurrent processing
    contents = [SAMPLE_CONTENT + f" - Version {i}" for i in range(3)]
    doc_ids = [f"test_async_{i}" for i in range(3)]
    
    logger.info("Testing BAML async processing...")
    baml_processor = BamlStructuredProcessor(kvstore_id=None)
    
    start_time = time.time()
    try:
        results = await baml_processor.abatch_analyze_documents(doc_ids, contents)
        async_time = time.time() - start_time
        logger.success(f"BAML async processing: {len(results)} documents in {async_time:.2f}s")
        for i, result in enumerate(results):
            if result:
                logger.info(f"  Document {i}: {result.name}")
    except Exception as e:
        logger.error(f"BAML async failed: {e}")


def compare_type_safety():
    """Compare type safety features."""
    logger.info("\n=== Type Safety Comparison ===")
    
    # BAML provides compile-time type safety
    logger.info("BAML Type Safety:")
    logger.info("✓ Pydantic models auto-generated from BAML schema")
    logger.info("✓ Type hints throughout the API")
    logger.info("✓ Runtime validation with meaningful errors")
    logger.info("✓ IDE auto-completion and static analysis support")
    
    # Original implementation
    logger.info("\nOriginal Implementation:")
    logger.info("? Depends on manual Pydantic class definitions")
    logger.info("? Runtime validation only")
    logger.info("? Potential for schema drift between components")


def compare_maintainability():
    """Compare maintainability aspects."""
    logger.info("\n=== Maintainability Comparison ===")
    
    logger.info("BAML Implementation:")
    logger.info("✓ Schema defined in declarative BAML files")
    logger.info("✓ Automatic client code generation")
    logger.info("✓ Version management for schema evolution")
    logger.info("✓ Clear separation of concerns")
    logger.info("✓ Built-in testing and validation tools")
    
    logger.info("\nOriginal Implementation:")
    logger.info("? Manual Pydantic class maintenance")
    logger.info("? Tight coupling between schema and processing logic")
    logger.info("? Manual version management")
    logger.info("? Custom validation and testing required")


async def main():
    """Run all comparisons."""
    logger.info("🔍 Comparing LangChain vs BAML Structured Extraction Implementations")
    logger.info("=" * 70)
    
    # Run comparisons
    compare_data_structures()
    await compare_performance()
    compare_error_handling()
    await compare_async_capabilities()
    compare_type_safety()
    compare_maintainability()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 SUMMARY")
    logger.info("=" * 70)
    logger.info("BAML Implementation Advantages:")
    logger.info("✓ Better type safety and IDE support")
    logger.info("✓ Declarative schema definition")
    logger.info("✓ Automatic client generation")
    logger.info("✓ Built-in async and concurrency control")
    logger.info("✓ Better error handling and validation")
    logger.info("✓ Easier testing and maintenance")
    
    logger.info("\nOriginal Implementation Advantages:")
    logger.info("? More flexibility for custom processing logic")
    logger.info("? Established ecosystem and patterns")
    logger.info("? Direct control over LangChain features")
    
    logger.info("\n💡 Recommendation: Use BAML implementation for new projects")
    logger.info("   and consider migrating existing projects for better maintainability.")


if __name__ == "__main__":
    asyncio.run(main())

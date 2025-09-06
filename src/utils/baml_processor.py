"""BAML-based structured data processor for documents.

This module provides the BamlStructuredProcessor class that uses BAML for extracting
structured data from documents. It supports both synchronous and asynchronous processing
with caching capabilities.

Key Features:
    - Uses BAML's ExtractRainbow function for structured data extraction
    - Supports both sync and async document processing
    - Integrates with KV store for caching results
    - Handles concurrent processing with semaphore control

Usage Examples:
    ```python
    from src.utils.baml_processor import BamlStructuredProcessor
    
    # Process a single document
    processor = BamlStructuredProcessor(kvstore_id="file")
    result = processor.analyze_document("doc1", markdown_content)
    
    # Process multiple files
    await processor.process_files(md_files, batch_size=5)
    ```
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import nest_asyncio
from loguru import logger
from upath import UPath

from src.demos.ekg.baml_client.async_client import b as baml_async_client
from src.demos.ekg.baml_client.types import ReviewedOpportunity
from src.utils.pydantic.kv_store import PydanticStore, save_object_to_kvstore


class BamlStructuredProcessor:
    """Processor that uses BAML for extracting structured data from documents."""

    def __init__(self, kvstore_id: str | None = None):
        self.kvstore_id = kvstore_id or "file"

    async def abatch_analyze_documents(
        self, document_ids: list[str], markdown_contents: list[str]
    ) -> list[ReviewedOpportunity]:
        """Process multiple documents asynchronously with caching using BAML."""
        analyzed_docs: list[ReviewedOpportunity] = []
        remaining_ids: list[str] = []
        remaining_contents: list[str] = []

        # Check cache first
        if self.kvstore_id:
            for doc_id, content in zip(document_ids, markdown_contents, strict=True):
                cached_doc = PydanticStore(kvstore_id=self.kvstore_id, model=ReviewedOpportunity).load_object(doc_id)

                if cached_doc:
                    analyzed_docs.append(cached_doc)
                    logger.info(f"Loaded cached document: {doc_id}")
                else:
                    remaining_ids.append(doc_id)
                    remaining_contents.append(content)
        else:
            remaining_ids = document_ids
            remaining_contents = markdown_contents

        if not remaining_ids:
            return analyzed_docs

        # Process uncached documents using BAML async client
        logger.info(f"Processing {len(remaining_ids)} documents with BAML async client...")

        # Process documents with async concurrency for better performance
        async def process_single_document(doc_id: str, content: str) -> ReviewedOpportunity | None:
            try:
                # Use BAML's async ExtractRainbow function
                result = await baml_async_client.ExtractRainbow(rainbow_file=content)

                # Add document_id as a custom attribute
                result_dict = result.model_dump()
                result_dict["document_id"] = doc_id
                result_with_id = ReviewedOpportunity(**result_dict)

                logger.success(f"Processed document: {doc_id}")

                # Save to KV store
                if self.kvstore_id:
                    save_object_to_kvstore(doc_id, result_with_id, kv_store_id=self.kvstore_id)
                    logger.debug(f"Saved to KV store: {doc_id}")

                return result_with_id

            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                return None

        # Process all documents concurrently with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def process_with_semaphore(doc_id: str, content: str) -> ReviewedOpportunity | None:
            async with semaphore:
                return await process_single_document(doc_id, content)

        tasks = [
            process_with_semaphore(doc_id, content)
            for doc_id, content in zip(remaining_ids, remaining_contents, strict=True)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        for result in results:
            if isinstance(result, ReviewedOpportunity):
                analyzed_docs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")

        return analyzed_docs

    def analyze_document(self, document_id: str, markdown: str) -> ReviewedOpportunity:
        """Analyze a single document synchronously using BAML."""
        try:
            # Try to use the current event loop if available
            loop = asyncio.get_running_loop()
            # If we're in an async context, use nest_asyncio
            nest_asyncio.apply()
            results = loop.run_until_complete(self.abatch_analyze_documents([document_id], [markdown]))
        except RuntimeError:
            # No event loop running, use asyncio.run
            try:
                results = asyncio.run(self.abatch_analyze_documents([document_id], [markdown]))
            except RuntimeError:
                # Last resort: create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self.abatch_analyze_documents([document_id], [markdown]))
                loop.close()

        if results:
            return results[0]
        else:
            raise ValueError(f"Failed to process document: {document_id}")

    async def process_files(self, md_files: list[UPath], batch_size: int = 5) -> None:
        """Process markdown files in batches using BAML."""
        document_ids = []
        markdown_contents = []
        valid_files = []

        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                document_ids.append(file_path.stem)
                markdown_contents.append(content)
                valid_files.append(file_path)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not document_ids:
            logger.warning("No valid files to process")
            return

        logger.info(f"Processing {len(valid_files)} files using BAML. Output in '{self.kvstore_id}' KV Store")

        # Process all documents (BAML handles batching internally)
        _ = await self.abatch_analyze_documents(document_ids, markdown_contents)


# Sample CV for testing
SAMPLE_CV = """# John Doe - Senior Software Engineer

## Contact Information
- Email: john.doe@email.com
- Phone: +1-555-123-4567
- LinkedIn: linkedin.com/in/johndoe
- Location: San Francisco, CA

## Professional Summary
Experienced software engineer with 8+ years developing scalable web applications
and distributed systems. Strong background in Python, JavaScript, and cloud
technologies. Passionate about clean code and agile development practices.

## Work Experience

### Senior Software Engineer - TechCorp Inc. (2020-Present)
- Led development of microservices architecture serving 1M+ users daily
- Implemented CI/CD pipelines reducing deployment time by 70%
- Mentored junior developers and conducted code reviews

### Software Engineer - StartupXYZ (2017-2020)
- Built RESTful APIs using Python and Django
- Developed React-based frontend applications
- Optimized database queries improving performance by 40%

## Education
- B.S. Computer Science, University of California, Berkeley (2017)
- GPA: 3.8/4.0

## Skills
- Programming: Python, JavaScript, TypeScript, Java
- Frameworks: Django, Flask, React, Node.js
- Cloud: AWS, Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, Redis
"""

if __name__ == "__main__":
    # Quick test with CV extraction
    logger.info("Running BAML processor test with sample CV...")
    
    processor = BamlStructuredProcessor(kvstore_id="memory")
    
    try:
        result = processor.analyze_document("test_cv", SAMPLE_CV)
        logger.success("CV extraction successful!")
        logger.info(f"Extracted data: {result.model_dump_json(indent=2)}")
    except Exception as e:
        logger.error(f"CV extraction failed: {e}")
        raise

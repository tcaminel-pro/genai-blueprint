"""Mistral OCR integration for document processing.

Provides tools for extracting text from PDFs using Mistral's OCR API, with support for:
- Single file processing
- Batch processing
- Caching results
- Integration with LangChain document loaders

The module handles both local files and remote URLs, and can process documents
asynchronously in batches for improved performance.
"""

import asyncio
import base64
import json
import os
from typing import Iterator

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from mistralai import Mistral
from mistralai.models import OCRResponse
from rich.progress import Progress, SpinnerColumn, TextColumn
from upath import UPath

from src.utils.pydantic.kv_store import load_object_from_kvstore, save_object_to_kvstore


def _encode_to_base64(path: UPath) -> str:
    """Encode file content to base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# taken from https://docs.mistral.ai/capabilities/document/#document-ocr-processor
# TODO : Impletent Asnyc


def mistral_ocr(path: UPath, use_cache: bool = True) -> OCRResponse:
    """Process a PDF file using Mistral OCR API.

    Handles both local files and remote URLs. Supports caching of results
    to avoid repeated processing of the same file.

    Args:
        path: Path to PDF file (local or remote)
        use_cache: Whether to use cached results if available

    Returns:
        OCRResponse containing extracted text and metadata
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")
    client = Mistral(api_key=api_key)

    if use_cache:
        cached_ocr = load_object_from_kvstore(model_class=OCRResponse, key=str(path))
        if cached_ocr:
            logger.info(f"use cached OCR for: '{str(path)}'")
            return cached_ocr
    if path.protocol in ["http", "https"]:
        document_url = str(path)
    else:
        base64_file = _encode_to_base64(path)
        document_url = f"data:application/pdf;base64,{base64_file}"

    #     uploaded_pdf = client.files.upload(
    #         file={
    #             "file_name": "uploaded_file.pdf",
    #             "content": open("uploaded_file.pdf", "rb"),
    #         },
    #         purpose="ocr",
    #     )
    #     retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
    #     signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    #     ocr_response = client.ocr.process(
    #     model="mistral-ocr-latest",
    #     document={
    #         "type": "document_url",
    #         "document_url": signed_url.url,
    #     }
    # )

    logger.info(f"Call Mistral OCR for:'{str(path)}'")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": document_url},
    )
    if use_cache:
        save_object_to_kvstore(key=str(path), obj=ocr_response)
    return ocr_response


class MistralOcrLoader(BaseLoader):
    """Load PDF documents using Mistral's OCR Model and API.

    Args:
        path: Path to the PDF file (local or remote)
        use_cache: Whether to use cached OCR results
    """

    def __init__(self, path: UPath | str, use_cache: bool = True) -> None:
        self.path = path
        self.use_cache = use_cache

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load document content using Mistral OCR."""
        if isinstance(self.path, str):
            self.path = UPath(self.path)
        ocr_response = mistral_ocr(self.path, self.use_cache)
        for page in ocr_response.pages:
            yield Document(
                page_content=page.markdown,
                metadata={"source": str(self.path), "index": page.index},
            )


def create_batch_file(document_urls: list[str], output_file: str) -> None:
    """Create a JSONL batch file for Mistral OCR batch processing.

    Args:
        document_urls: List of document URLs to process
        output_file: Path to save the batch file
    """
    """Create a batch file for Mistral OCR processing.

    Args:
        document_urls: List of document URLs (can be data URLs with base64 encoded content)
        output_file: Path to output JSONL batch file
    """
    with open(output_file, "w") as file:
        for index, url in enumerate(document_urls):
            entry = {"custom_id": str(index), "body": {"document": {"type": "document_url", "document_url": url}}}
            file.write(json.dumps(entry) + "\n")


def sanitize_filename(filename: str) -> str:
    """Replace whitespace characters in filename with underscores."""
    return filename.replace(" ", "_")


async def process_pdf_batch(pdf_paths: list[UPath], output_dir: UPath, use_cache: bool = True) -> None:
    """Process multiple PDF files using Mistral's batch OCR API.

    Handles caching, progress tracking, and result saving. Processes files
    asynchronously for better performance with large batches.

    Args:
        pdf_paths: List of PDF files to process
        output_dir: Directory to save extracted text files
        use_cache: Whether to use cached results if available
    """

    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")

    client = Mistral(api_key=api_key)
    ocr_model = "mistral-ocr-latest"

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Process files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Preparing PDF files...", total=len(pdf_paths))

        # First check cache for all files if enabled
        if use_cache:
            cached_files = []
            for i, pdf_path in enumerate(pdf_paths):
                progress.update(
                    task, description=f"[cyan]Checking cache for {pdf_path.name} ({i + 1}/{len(pdf_paths)})"
                )
                cached_ocr = load_object_from_kvstore(model_class=OCRResponse, key=str(pdf_path))
                if cached_ocr:
                    logger.info(f"Using cached OCR for: '{str(pdf_path)}'")
                    # Save to output directory with sanitized filename
                    safe_filename = sanitize_filename(f"{pdf_path.stem}.md")
                    output_file = output_dir / safe_filename
                    with open(output_file, "w") as f:
                        for page in cached_ocr.pages:
                            f.write(f"## Page {page.index + 1}\n\n")
                            f.write(page.markdown)
                            f.write("\n\n")
                    cached_files.append(pdf_path)
                progress.advance(task)

            # Remove cached files from processing list
            pdf_paths = [path for path in pdf_paths if path not in cached_files]

            if not pdf_paths:
                logger.info("All files were found in cache. No need for batch processing.")
                return

        # Create document URLs for remaining files
        progress.update(task, description="[cyan]Preparing batch file...")
        document_urls = []
        for pdf_path in pdf_paths:
            if pdf_path.protocol in ["http", "https"]:
                document_url = str(pdf_path)
            else:
                base64_file = _encode_to_base64(pdf_path)
                document_url = f"data:application/pdf;base64,{base64_file}"
            document_urls.append(document_url)

        # Create batch file
        batch_file_path = "batch_ocr_file.jsonl"
        #        debug(document_urls)
        create_batch_file(document_urls, batch_file_path)
        progress.update(task, completed=len(pdf_paths), description="[cyan]Batch file created")

        # Upload batch file
        progress.update(task, description="[cyan]Uploading batch file...")
        batch_data = client.files.upload(
            file={"file_name": batch_file_path, "content": open(batch_file_path, "rb")}, purpose="batch"
        )

        # Create batch job
        progress.update(task, description="[cyan]Creating batch job...")
        created_job = client.batch.jobs.create(
            input_files=[batch_data.id], model=ocr_model, endpoint="/v1/ocr", metadata={"job_type": "pdf_ocr_batch"}
        )

        # Monitor job progress
        retrieved_job = client.batch.jobs.get(job_id=created_job.id)
        monitor_task = progress.add_task(f"[green]Batch job status: {retrieved_job.status}", total=len(pdf_paths))

        while retrieved_job.status in ["QUEUED", "RUNNING"]:
            retrieved_job = client.batch.jobs.get(job_id=created_job.id)
            progress.update(
                monitor_task,
                completed=retrieved_job.succeeded_requests + retrieved_job.failed_requests,
                description=f"[green]Batch job status: {retrieved_job.status} - "
                + f"Completed: {retrieved_job.succeeded_requests + retrieved_job.failed_requests}/{retrieved_job.total_requests} "
                + f"({round((retrieved_job.succeeded_requests + retrieved_job.failed_requests) / retrieved_job.total_requests * 100, 1)}%)",
            )
            await asyncio.sleep(2)

        # Download results
        if retrieved_job.status == "SUCCESS" and retrieved_job.output_file:
            progress.update(monitor_task, description="[green]Downloading results...")
            response = client.files.download(file_id=retrieved_job.output_file)

            # Read the response content
            response_content = response.read().decode("utf-8")

            # Process the results
            results_task = progress.add_task("[blue]Processing results...", total=len(pdf_paths))

            # Parse the JSONL response
            results = response_content.strip().split("\n")
            for i, result_line in enumerate(results):
                progress.update(results_task, description=f"[blue]Processing result {i + 1}/{len(results)}")

                result = json.loads(result_line)
                pdf_path = pdf_paths[int(result["custom_id"])]

                # The batch API returns a different structure than the direct API
                # We need to extract the actual OCR response from the batch result
                response_data = result["response"]

                # Check if the response contains the actual OCR data or is wrapped
                if "body" in response_data and isinstance(response_data["body"], dict):
                    ocr_data = response_data["body"]
                    ocr_response = OCRResponse.model_validate(ocr_data)
                else:
                    logger.warning(f"Unexpected response format for {pdf_path.name}. Trying to adapt...")
                    # Try to construct an OCR response from available data
                    if "pages" in response_data:
                        ocr_response = OCRResponse.model_validate(response_data)
                    else:
                        logger.error(f"Could not parse OCR response for {pdf_path.name}")
                        progress.advance(results_task)
                        continue

                # Cache the result
                if use_cache:
                    save_object_to_kvstore(key=str(pdf_path), obj=ocr_response)

                # Save to output directory with sanitized filename
                safe_filename = sanitize_filename(f"{pdf_path.stem}.md")
                output_file = output_dir / safe_filename
                with open(output_file, "w") as f:
                    for page in ocr_response.pages:
                        f.write(f"## Page {page.index + 1}\n\n")
                        f.write(page.markdown)
                        f.write("\n\n")

                progress.advance(results_task)

            # Clean up the batch file
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

            logger.info(f"Batch processing complete. Results saved to {output_dir}")
        elif retrieved_job.status == "SUCCESS" and not retrieved_job.output_file:
            logger.warning("Batch job completed successfully but no output file was generated.")
            # This can happen if all files were processed but had errors
            for pdf_path in pdf_paths:
                logger.error(f"Failed to process: {pdf_path}")
        else:
            logger.error(f"Batch job failed with status: {retrieved_job.status}")


# Quick test
if __name__ == "__main__":
    from devtools import debug

    doc = UPath("https://arxiv.org/pdf/2201.04234")

    # res = mistral_ocr(doc, use_cache=True)
    # debug(res)
    loader = MistralOcrLoader(doc)
    documents = loader.load()  # or use lazy_load() for streaming
    debug(documents)

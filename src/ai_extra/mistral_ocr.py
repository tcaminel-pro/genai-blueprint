import asyncio
import base64
import json
import os
from pathlib import Path as StdPath
from typing import Iterator, List, Optional

import typer
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from mistralai import Mistral
from mistralai.async_client import MistralAsyncClient
from mistralai.models import OCRResponse
from rich.progress import Progress, SpinnerColumn, TextColumn
from trio import Path
from upath import UPath

from src.utils.pydantic.kv_store import load_object_from_kvstore, save_object_to_kvstore


def encode_to_base64(path: UPath) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# taken from https://docs.mistral.ai/capabilities/document/#document-ocr-processor
# TODO : Impletent Asnyc and Batch


def mistral_ocr(path: UPath, use_cache: bool = True) -> OCRResponse:
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
        base64_file = encode_to_base64(path)
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

def create_batch_file(pdf_paths: list[UPath], output_file: Path) -> None:
    """Create a batch file for Mistral OCR processing.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_file: Path to output JSONL batch file
    """
    with open(output_file, 'w') as file:
        for index, path in enumerate(pdf_paths):
            if path.protocol in ["http", "https"]:
                document_url = str(path)
            else:
                base64_file = encode_to_base64(path)
                document_url = f"data:application/pdf;base64,{base64_file}"
                
            entry = {
                "custom_id": str(index),
                "body": {
                    "document": {
                        "type": "document_url",
                        "document_url": document_url
                    }
                }
            }
            file.write(json.dumps(entry) + '\n')


async def process_pdf_batch(
    pdf_paths: list[UPath], 
    output_dir: UPath,
    use_cache: bool = True
) -> None:
    """Process a batch of PDF files using Mistral OCR asynchronously.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_dir: Directory to save OCR results
        use_cache: Whether to use cached OCR results
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")
    
    client = MistralAsyncClient(api_key=api_key)
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Process files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Processing PDF files...", total=len(pdf_paths))
        
        for i, pdf_path in enumerate(pdf_paths):
            progress.update(task, description=f"[cyan]Processing {pdf_path.name} ({i+1}/{len(pdf_paths)})")
            
            # Check cache first if enabled
            if use_cache:
                cached_ocr = load_object_from_kvstore(model_class=OCRResponse, key=str(pdf_path))
                if cached_ocr:
                    logger.info(f"Using cached OCR for: '{str(pdf_path)}'")
                    # Save to output directory
                    output_file = output_dir / f"{pdf_path.stem}.md"
                    with open(output_file, 'w') as f:
                        for page in cached_ocr.pages:
                            f.write(f"## Page {page.index + 1}\n\n")
                            f.write(page.markdown)
                            f.write("\n\n")
                    progress.advance(task)
                    continue
            
            # Process with Mistral OCR
            try:
                if pdf_path.protocol in ["http", "https"]:
                    document_url = str(pdf_path)
                else:
                    base64_file = encode_to_base64(pdf_path)
                    document_url = f"data:application/pdf;base64,{base64_file}"
                
                ocr_response = await client.ocr.process(
                    model="mistral-ocr-latest",
                    document={"type": "document_url", "document_url": document_url},
                )
                
                # Cache the result
                if use_cache:
                    save_object_to_kvstore(key=str(pdf_path), obj=ocr_response)
                
                # Save to output directory
                output_file = output_dir / f"{pdf_path.stem}.md"
                with open(output_file, 'w') as f:
                    for page in ocr_response.pages:
                        f.write(f"## Page {page.index + 1}\n\n")
                        f.write(page.markdown)
                        f.write("\n\n")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
            
            progress.advance(task)


# CLI command
app = typer.Typer()

@app.command()
def ocr_pdf(
    file_patterns: List[str] = typer.Argument(..., help="File patterns to match PDF files (glob patterns)"),
    output_dir: str = typer.Option("./ocr_output", help="Directory to save OCR results"),
    use_cache: bool = typer.Option(True, help="Use cached OCR results if available"),
    recursive: bool = typer.Option(False, help="Search for files recursively")
):
    """Process PDF files with Mistral OCR and save the results as markdown files.
    
    Example:
        python -m src.ai_extra.mistral_ocr ocr_pdf "*.pdf" "data/*.pdf" --output-dir=./ocr_results
    """
    # Collect all PDF files matching the patterns
    all_files = []
    for pattern in file_patterns:
        path = UPath(pattern)
        
        # Handle glob patterns
        if "*" in pattern:
            base_dir = path.parent
            if recursive:
                matched_files = list(base_dir.glob(f"**/{path.name}"))
            else:
                matched_files = list(base_dir.glob(path.name))
            all_files.extend(matched_files)
        else:
            # Direct file path
            if path.exists():
                all_files.append(path)
    
    # Filter for PDF files
    pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf']
    
    if not pdf_files:
        logger.warning("No PDF files found matching the provided patterns.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process the files
    output_path = UPath(output_dir)
    asyncio.run(process_pdf_batch(pdf_files, output_path, use_cache))
    
    logger.info(f"OCR processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    app()

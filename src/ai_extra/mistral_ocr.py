import base64
import json
import os
from typing import Iterator

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from mistralai import Mistral
from mistralai.models import OCRResponse
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

# The following code is taken from an Notebook example of processing images in batch
# Correct it to make it work with pdf files, and asynchrously so it could be called
# from a CLI commnand.
# Also write the CLI command using Typer, that takes pattern of files and output dir as input
# and process the files with regulat message to display the  advancement
# AI!
def create_batch_file(image_urls: list[UPath], output_file: Path) -> None:
    with open(output_file, 'w') as file:
        for index, url in enumerate(image_urls):
            entry = {
                "custom_id": str(index),
                "body": {
                    "document": {
                        "type": "image_url",
                        "image_url": url
                    },
                    "include_image_base64": True
                }
            }
            file.write(json.dumps(entry) + '\n')


def encode_file (files : list[UPath]) : 
    image_urls = []
    for file in files:
        # Encode the image data to base64 and add the url to the list
        base64_image = encode_image_data(image_data)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        image_urls.append(image_url)
    batch_file = "batch_file.jsonl"
    create_batch_file(image_urls, batch_file)
    import time
    from IPython.display import clear_output

    while retrieved_job.status in ["QUEUED", "RUNNING"]:
        retrieved_job = client.batch.jobs.get(job_id=created_job.id)

        clear_output(wait=True)  # Clear the previous output ( User Friendly )
        print(f"Status: {retrieved_job.status}")
        print(f"Total requests: {retrieved_job.total_requests}")
        print(f"Failed requests: {retrieved_job.failed_requests}")
        print(f"Successful requests: {retrieved_job.succeeded_requests}")
        print(
            f"Percent done: {round((retrieved_job.succeeded_requests + retrieved_job.failed_requests) / retrieved_job.total_requests, 4) * 100}%"
        )
        time.sleep(2)

if __name__ == "__main__":
    from devtools import debug

    doc = UPath("https://arxiv.org/pdf/2201.04234")

    # res = mistral_ocr(doc, use_cache=True)
    # debug(res)
    loader = MistralOcrLoader(doc)
    documents = loader.load()  # or use lazy_load() for streaming
    debug(documents)

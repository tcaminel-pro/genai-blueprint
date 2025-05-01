import base64
import os

from devtools import debug
from loguru import logger
from mistralai import Mistral
from mistralai.models import OCRResponse
from upath import UPath
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders.base import BaseLoader

from src.utils.pydantic.kv_store import read_pydantic_from_store, save_pydantic_to_store


def encode_to_base64(path: UPath) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# taken from https://docs.mistral.ai/capabilities/document/#document-ocr-processor


async def mistral_ocr(path: UPath, use_cache: bool = True) -> OCRResponse:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")
    client = Mistral(api_key=api_key)

    if use_cache:
        cached_ocr = read_pydantic_from_store(model_class=OCRResponse, key=str(path))
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
    ocr_response = await client.ocr.process_async(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": document_url},
    )
    if use_cache:
        save_pydantic_to_store(key=str(path), obj=ocr_response)
    return ocr_response


# def pdf_query_message(param_dict: dict, config: dict) -> list[BaseMessage]:
# HumanMessage(
#     [
#         {
#             "type": "media",
#             "data": "pdf_base64",
#             "mime_type": "application/pdf",
#         },
#         "What's the first page of the pdf?",
#     ]
# ),

class MistralOcrLoader(BaseLoader) -> Interator[Document]:
    """`OCR using Mistral API"""



if __name__ == "__main__":
    doc = UPath("https://arxiv.org/pdf/2201.04234")

    res = mistral_ocr(doc, use_cache=True)
    debug(res)

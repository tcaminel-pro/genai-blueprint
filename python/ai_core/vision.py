"""Vision processing and image analysis utilities.

This module provides tools for creating multimodal messages for AI image analysis,
supporting local files, remote URLs, and vision-enabled language models.

Key Features:
- Flexible image input (local/remote)
- Customizable system prompts
- Structured output generation
"""

import base64
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage


def image_query_message(param_dict: dict, config: dict) -> list[BaseMessage]:
    """Create a multimodal message for AI image analysis.

    Prepares messages for vision-enabled language models with support for 
    local and remote images, custom queries, and optional output parsing.

    Args:
        param_dict (dict): Query parameters including 'query', 'image_paths', 
                           optional 'system' and 'output_parser'
        config (dict): Configuration dictionary

    Returns:
        list[BaseMessage]: Messages ready for a multimodal AI model

    Raises:
        ValueError: For invalid parameters or image paths
    """
    allowed_params = ["query", "image_paths", "output_parser", "system"]
    for key in param_dict:
        if key not in allowed_params:
            raise ValueError(f"Unexpected parameter: {key}")

    query = param_dict["query"]
    image_paths = param_dict["image_paths"]
    system_message = (
        param_dict.get("system")
        or "You are a helpful assistant that analyse images and answer questions provided by the user."
    )
    parser = param_dict.get("output_parser")

    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    human_messages = [{"type": "text", "text": query}]
    if parser:
        human_messages.append({"type": "text", "text": parser.get_format_instructions()})

    # Nova LLM (and probably others) have a different way to deal with image
    # see https://docs.aws.amazon.com/pdfs/nova/latest/userguide/nova-ug.pdf
    nova_api = (llm_id := config["metadata"].get("llm_id")) and str(llm_id).startswith("nova")
    if nova_api:
        raise NotImplementedError("Nova LLM support not implemented")

    for url_or_filename in image_paths:
        scheme = urlparse(str(url_or_filename)).scheme
        if isinstance(url_or_filename, Path) or scheme == "":
            with open(url_or_filename, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                url = f"data:image/png;base64,{image_data}"
        elif str(scheme) in ["http", "https"]:
            url = url_or_filename
        else:
            raise ValueError(f"unknown image type or path: '{url_or_filename}'")

        human_messages.append({"type": "image_url", "image_url": {"url": url}})

    messages = [SystemMessage(content=system_message), HumanMessage(content=human_messages)]  # type: ignore
    return messages

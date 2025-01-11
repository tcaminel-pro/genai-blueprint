"""
Vision processing and image analysis utilities.

This module provides tools for integrating image analysis capabilities into AI workflows,
supporting both local and remote image sources. It creates multimodal messages compatible
with vision-enabled language models.

Key Features:
- Support for local files and remote URLs
- Structured output generation
- Customizable system prompts
- Integration with output parsers
- Base64 encoding for local images

Example:
    >>> # Analyze image with structured output
    >>> messages = image_query_message({
    ...     "query": "Describe the image",
    ...     "image_paths": "path/to/image.jpg",
    ...     "output_parser": MyOutputParser()
    ... })

    >>> # Send to vision-enabled LLM
    >>> response = llm.invoke(messages)
"""

import base64
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage


def image_query_message(param_dict: dict, config: dict) -> list[BaseMessage]:
    """
    Create a multimodal message for AI images analysis with optional structured output

    This function prepares a message suitable for vision-enabled language models,
    supporting both local image files and image URLs. It can include a custom
    system message and optional output parsing instructions.

    Args:
        param_dict (dict): A dictionary containing query parameters with these possible keys:
            - 'query' (str): The main text query about the image(s)
            - 'image_paths' (str or list): Path(s) to local image files or image URLs
            - 'system' (str, optional): Custom system message for context setting
            - 'output_parser' (object, optional): Langchain OutputParser object

    Returns:
        list[BaseMessage]: A list of messages ready for a multimodal AI model

    Raises:
        ValueError: If unexpected parameters are provided or image paths are invalid

    Example :
    .. code-block:: python
        class ImageDesc(BaseModel):
            background: str = Field(description="image background")
            animals: list[str] = Field(description="Animals found in the image")
            actions: str = Field(description="What are the animals doing")

        parser = PydanticOutputParser(pydantic_object=ImageDesc)
        image_url = "https://example.com/image.jpg"
        chain = image_query_message | llm | parser  # ignore
        response = chain.invoke({"query": "describe that image in JSON", "image_paths": image_url, "output_parser": parser})
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

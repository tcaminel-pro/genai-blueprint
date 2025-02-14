from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.ai_core.vision import image_query_message


def test_image_query_message_local_image(tmp_path):
    # Create a temporary image file
    test_image = tmp_path / "test_image.png"
    test_image.write_bytes(b"fake image content")

    param_dict = {"query": "What's in this image?", "image_paths": [test_image]}
    config = {"metadata": {}}

    messages = image_query_message(param_dict, config)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    # Check human message content
    human_message_content = messages[1].content
    assert isinstance(human_message_content, list)
    assert len(human_message_content) == 2  # query + image
    assert human_message_content[0]["type"] == "text"
    assert human_message_content[0]["text"] == "What's in this image?"
    assert human_message_content[1]["type"] == "image_url"
    assert human_message_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_query_message_remote_image():
    param_dict = {"query": "Describe this remote image", "image_paths": ["https://example.com/image.jpg"]}
    config = {"metadata": {}}

    messages = image_query_message(param_dict, config)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    human_message_content = messages[1].content
    assert isinstance(human_message_content, list)
    assert len(human_message_content) == 2  # query + image
    assert human_message_content[0]["type"] == "text"
    assert human_message_content[0]["text"] == "Describe this remote image"
    assert human_message_content[1]["type"] == "image_url"
    assert human_message_content[1]["image_url"]["url"] == "https://example.com/image.jpg"


def test_image_query_message_invalid_parameter():
    with pytest.raises(ValueError, match="Unexpected parameter"):
        image_query_message({"invalid_param": "test"}, {"metadata": {}})


def test_image_query_message_with_output_parser():
    mock_parser = Mock()
    mock_parser.get_format_instructions.return_value = "Format: JSON"

    param_dict = {
        "query": "Parse this image",
        "image_paths": ["https://example.com/image.jpg"],
        "output_parser": mock_parser,
    }
    config = {"metadata": {}}

    messages = image_query_message(param_dict, config)

    human_message_content = messages[1].content
    assert len(human_message_content) == 3  # query + parser instructions + image
    assert human_message_content[1]["type"] == "text"
    assert human_message_content[1]["text"] == "Format: JSON"


def test_image_query_message_nova_api():
    with pytest.raises(NotImplementedError, match="Nova LLM support not implemented"):
        image_query_message(
            {"query": "Test Nova API", "image_paths": ["https://example.com/image.jpg"]},
            {"metadata": {"llm_id": "nova-model"}},
        )

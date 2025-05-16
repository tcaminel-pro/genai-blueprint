"""Tests for MCP (Multi-Component Platform) client functionality.

This module contains unit tests for:
- MCP server configuration parsing
- Command alias handling
- Agent runner functionality

The tests verify that:
- Server configurations are correctly parsed from application config
- Command aliases (like uvx) are properly converted
- The agent runner correctly initializes and interacts with MCP tools
"""

import pytest
from langchain_mcp_adapters.client import StdioServerParameters
from pydantic import ValidationError

from src.ai_core.mcp_client import get_mcp_servers_dict


def test_get_mcp_servers_from_config_valid(monkeypatch):
    """Test that valid MCP server configurations are parsed correctly."""
    # Mock the global config to return test data
    test_config = {
        "testServer": {
            "command": "uv",
            "args": ["tool", "run", "test@1.0.0"],
            "env": {"TEST_VAR": "value"},
            "transport": "stdio",
        }
    }

    monkeypatch.setattr(
        "src.utils.config_mngr.global_config",
        lambda: type("MockConfig", (), {"get_dict": lambda self, key: test_config})(),
    )

    servers = get_mcp_servers_dict()
    assert "testServer" in servers
    assert servers["testServer"]["command"] == "uv"
    assert servers["testServer"]["args"] == ["tool", "run", "test@1.0.0"]
    assert servers["testServer"]["env"]["PATH"]  # PATH should be added
    assert servers["testServer"]["env"]["TEST_VAR"] == "value"


def test_get_mcp_servers_from_config_invalid():
    """Test that invalid MCP server configurations raise appropriate errors."""
    with pytest.raises(ValidationError):
        # Missing required fields
        StdioServerParameters(command=None, args=[])


def test_uvx_alias_handling(monkeypatch):
    """Test that uvx alias is properly converted to uv command."""
    test_config = {"testServer": {"command": "uvx", "args": ["test@1.0.0"], "transport": "stdio"}}

    monkeypatch.setattr(
        "src.utils.config_mngr.global_config",
        lambda: type("MockConfig", (), {"get_dict": lambda self, key: test_config})(),
    )

    servers = get_mcp_servers_dict()
    assert servers["testServer"]["command"] == "uv"
    assert servers["testServer"]["args"] == ["tool", "run", "test@1.0.0"]


@pytest.mark.asyncio
async def test_mcp_agent_runner(mocker):
    """Test the MCP agent runner with mocked components."""
    from langchain_core.language_models.chat_models import BaseChatModel

    from src.ai_core.mcp_client import mcp_agent_runner

    # Mock the model and server parameters
    mock_model = mocker.Mock(spec=BaseChatModel)
    mock_servers = [mocker.Mock(spec=StdioServerParameters)]

    # Mock the AsyncExitStack and MCPAdapt
    mock_exit_stack = mocker.patch("src.ai_extra.mcp_client.AsyncExitStack")
    mock_mcp_adapt = mocker.patch("src.ai_extra.mcp_client.MCPAdapt")

    # Call the function
    result = await mcp_agent_runner(mock_model, mock_servers, "test prompt")

    # Verify the interactions
    mock_exit_stack.return_value.__enter__.assert_called()
    mock_mcp_adapt.assert_called()
    assert result is not None

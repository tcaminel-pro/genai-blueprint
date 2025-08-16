"""Unit tests for MCP client functionality."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import StdioServerParameters

from src.ai_core.mcp_client import (
    dict_to_stdio_server_list,
    get_mcp_servers_dict,
    get_mcp_tools_info,
    update_server_parameters,
)


class TestUpdateServerParameters(unittest.TestCase):
    """Test cases for the update_server_parameters function."""

    def test_basic_server_parameters(self):
        """Test basic server parameter processing."""
        config = {"command": "echo", "args": ["hello", "world"], "transport": "stdio"}

        result = update_server_parameters(config)

        self.assertEqual(result["command"], "echo")
        self.assertEqual(result["args"], ["hello", "world"])
        self.assertEqual(result["transport"], "stdio")
        self.assertIn("PATH", result["env"])

    def test_uvx_command_alias(self):
        """Test that uvx command is properly aliased to uv tool run."""
        config = {"command": "uvx", "args": ["some-tool", "arg1"]}

        result = update_server_parameters(config)

        self.assertEqual(result["command"], "uv")
        self.assertEqual(result["args"], ["tool", "run", "some-tool", "arg1"])

    def test_default_transport(self):
        """Test that transport defaults to stdio if not specified."""
        config = {"command": "test", "args": []}

        result = update_server_parameters(config)

        self.assertEqual(result["transport"], "stdio")

    def test_environment_variables(self):
        """Test environment variable handling."""
        config = {"command": "test", "args": [], "env": {"CUSTOM_VAR": "value"}}

        result = update_server_parameters(config)

        self.assertIn("PATH", result["env"])
        self.assertEqual(result["env"]["CUSTOM_VAR"], "value")

    def test_removes_unused_keys(self):
        """Test that unused keys are removed from the configuration."""
        config = {
            "command": "test",
            "args": [],
            "description": "A test server",
            "example": "usage example",
            "disabled": False,
        }

        result = update_server_parameters(config)

        self.assertNotIn("description", result)
        self.assertNotIn("example", result)
        self.assertNotIn("disabled", result)


class TestDictToStdioServerList(unittest.TestCase):
    """Test cases for the dict_to_stdio_server_list function."""

    def test_empty_dict(self):
        """Test conversion of empty dictionary."""
        result = dict_to_stdio_server_list({})
        self.assertEqual(result, [])

    def test_single_server(self):
        """Test conversion of single server configuration."""
        servers = {"test_server": {"command": "echo", "args": ["hello"], "transport": "stdio"}}

        result = dict_to_stdio_server_list(servers)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], StdioServerParameters)
        self.assertEqual(result[0].command, "echo")
        self.assertEqual(result[0].args, ["hello"])

    def test_multiple_servers(self):
        """Test conversion of multiple server configurations."""
        servers = {
            "server1": {"command": "echo", "args": ["hello"], "transport": "stdio"},
            "server2": {"command": "cat", "args": ["file.txt"], "transport": "stdio"},
        }

        result = dict_to_stdio_server_list(servers)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], StdioServerParameters)
        self.assertIsInstance(result[1], StdioServerParameters)


class TestGetMcpServersDict(unittest.TestCase):
    """Test cases for the get_mcp_servers_dict function."""

    @patch("src.ai_core.mcp_client.global_config")
    @patch("src.ai_core.mcp_client.update_server_parameters")
    def test_get_all_servers(self, mock_update, mock_global_config):
        """Test retrieval of all configured servers."""
        mock_config = MagicMock()
        mock_config.merge_with.return_value.get_dict.return_value = {
            "server1": {"command": "test1"},
            "server2": {"command": "test2", "disabled": False},
            "server3": {"command": "test3", "disabled": True},
        }
        mock_global_config.return_value = mock_config
        mock_update.side_effect = lambda x: x

        result = get_mcp_servers_dict()

        self.assertIn("server1", result)
        self.assertIn("server2", result)
        self.assertNotIn("server3", result)
        mock_update.assert_called()

    @patch("src.ai_core.mcp_client.global_config")
    def test_filter_servers(self, mock_global_config):
        """Test filtering servers by name."""
        mock_config = MagicMock()
        mock_config.merge_with.return_value.get_dict.return_value = {
            "server1": {"command": "test1"},
            "server2": {"command": "test2"},
        }
        mock_global_config.return_value = mock_config

        result = get_mcp_servers_dict(filter=["server1"])

        self.assertEqual(list(result.keys()), ["server1"])

    @patch("src.ai_core.mcp_client.global_config")
    def test_missing_server_in_filter(self, mock_global_config):
        """Test error handling when filter contains non-existent server."""
        mock_config = MagicMock()
        mock_config.merge_with.return_value.get_dict.return_value = {"server1": {"command": "test1"}}
        mock_global_config.return_value = mock_config

        with self.assertRaises(ValueError) as cm:
            get_mcp_servers_dict(filter=["nonexistent"])

        self.assertIn("nonexistent", str(cm.exception))

    @patch("src.ai_core.mcp_client.global_config")
    @patch("src.ai_core.mcp_client.update_server_parameters")
    def test_server_configuration_error(self, mock_update, mock_global_config):
        """Test handling of server configuration errors."""
        mock_config = MagicMock()
        mock_config.merge_with.return_value.get_dict.return_value = {
            "server1": {"command": "test1"},
            "server2": {"command": "test2"},
        }
        mock_global_config.return_value = mock_config
        mock_update.side_effect = [Exception("config error"), {"command": "fixed"}]

        result = get_mcp_servers_dict()

        self.assertEqual(len(result), 1)
        self.assertIn("server2", result)


class TestMcpClientAsyncFunctions(unittest.IsolatedAsyncioTestCase):
    """Test cases for async functions in MCP client."""

    @patch("src.ai_core.mcp_client.stdio_client")
    @patch("src.ai_core.mcp_client.ClientSession")
    async def test_get_mcp_tools_info(self, mock_client_session, mock_stdio_client):
        """Test retrieval of tools information from MCP servers."""
        # Mock the client session and tools
        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        # Mock tools response - need to properly configure the mock objects
        mock_tools_response = MagicMock()
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Second tool"
        mock_tools_response.tools = [mock_tool1, mock_tool2]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

        # Mock stdio_client
        mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        # Mock get_mcp_servers_dict
        with patch("src.ai_core.mcp_client.get_mcp_servers_dict") as mock_get_servers:
            mock_get_servers.return_value = {"test_server": {"command": "test", "args": []}}

            result = await get_mcp_tools_info()

            self.assertIn("test_server", result)
            self.assertEqual(len(result["test_server"]), 2)
            self.assertEqual(result["test_server"]["tool1"], "First tool")
            self.assertEqual(result["test_server"]["tool2"], "Second tool")


if __name__ == "__main__":
    unittest.main()

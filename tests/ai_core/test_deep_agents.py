"""
Tests for Deep Agents Integration

This module contains comprehensive tests for the deepagents integration,
including factory methods, agent creation, tool management, and execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.ai_core.deep_agents import (
    DeepAgentConfig,
    SubAgentConfig,
    DeepAgentFactory,
    deep_agent_factory,
    create_deep_agent_from_config,
    create_research_deep_agent,
    create_coding_deep_agent,
    run_deep_agent
)


class TestDeepAgentConfig:
    """Test DeepAgentConfig model"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DeepAgentConfig()
        
        assert config.name == "Deep Agent"
        assert config.instructions == ""
        assert config.model is None
        assert config.builtin_tools is None
        assert config.on_duplicate_tools == "warn"
        assert config.subagents is None
        assert config.enable_file_system is True
        assert config.enable_planning is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = DeepAgentConfig(
            name="Test Agent",
            instructions="Test instructions",
            model="gpt-4",
            builtin_tools=["write_file", "read_file"],
            on_duplicate_tools="error",
            enable_file_system=False,
            enable_planning=False
        )
        
        assert config.name == "Test Agent"
        assert config.instructions == "Test instructions"
        assert config.model == "gpt-4"
        assert config.builtin_tools == ["write_file", "read_file"]
        assert config.on_duplicate_tools == "error"
        assert config.enable_file_system is False
        assert config.enable_planning is False
    
    def test_subagent_config(self):
        """Test SubAgentConfig model"""
        subagent = SubAgentConfig(
            name="test-subagent",
            description="Test subagent",
            prompt="You are a test subagent"
        )
        
        assert subagent.name == "test-subagent"
        assert subagent.description == "Test subagent"
        assert subagent.prompt == "You are a test subagent"
        assert subagent.tools is None
        assert subagent.model_settings is None


class TestDeepAgentFactory:
    """Test DeepAgentFactory class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.factory = DeepAgentFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization"""
        assert self.factory.agents == {}
        assert self.factory.default_model is None
    
    @patch('src.ai_core.deep_agents.get_llm')
    def test_set_default_model_none(self, mock_get_llm):
        """Test setting default model with None"""
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        
        self.factory.set_default_model(None)
        
        mock_get_llm.assert_called_once()
        assert self.factory.default_model == mock_model
    
    @patch('src.ai_core.deep_agents.get_llm')
    def test_set_default_model_string(self, mock_get_llm):
        """Test setting default model with string"""
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        
        self.factory.set_default_model("gpt-4")
        
        mock_get_llm.assert_called_once_with(model_name="gpt-4")
        assert self.factory.default_model == mock_model
    
    def test_set_default_model_object(self):
        """Test setting default model with object"""
        mock_model = MagicMock(spec=BaseChatModel)
        
        self.factory.set_default_model(mock_model)
        
        assert self.factory.default_model == mock_model
    
    def test_convert_langchain_tool(self):
        """Test converting LangChain tool to regular function"""
        # Create a mock LangChain tool
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.run.return_value = "tool result"
        
        # Convert the tool
        converted = self.factory._convert_langchain_tool(mock_tool)
        
        # Test the converted function
        assert callable(converted)
        assert converted.__name__ == "test_tool"
        assert converted.__doc__ == "Test tool description"
        
        # Test execution
        result = converted(test_param="value")
        mock_tool.run.assert_called_once_with({"test_param": "value"})
    
    def test_prepare_tools_mixed(self):
        """Test preparing mixed tools (LangChain and regular functions)"""
        # Create a mock LangChain tool
        mock_langchain_tool = MagicMock(spec=BaseTool)
        mock_langchain_tool.name = "langchain_tool"
        mock_langchain_tool.description = "LangChain tool"
        
        # Create a regular function
        def regular_tool():
            """Regular tool"""
            return "regular result"
        
        # Prepare tools
        tools = [mock_langchain_tool, regular_tool]
        prepared = self.factory._prepare_tools(tools)
        
        # Check results
        assert len(prepared) == 2
        assert callable(prepared[0])
        assert prepared[1] == regular_tool
    
    @patch('src.ai_core.deep_agents.create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_create_agent_basic(self, mock_get_llm, mock_create_deep_agent):
        """Test basic agent creation"""
        # Setup mocks
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent
        
        # Create config
        config = DeepAgentConfig(
            name="Test Agent",
            instructions="Test instructions"
        )
        
        # Create agent
        agent = self.factory.create_agent(config, tools=[], async_mode=False)
        
        # Verify
        mock_create_deep_agent.assert_called_once()
        assert self.factory.agents["Test Agent"] == mock_agent
        assert agent == mock_agent
    
    @patch('src.ai_core.deep_agents.async_create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_create_agent_async(self, mock_get_llm, mock_async_create_deep_agent):
        """Test async agent creation"""
        # Setup mocks
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_async_create_deep_agent.return_value = mock_agent
        
        # Create config
        config = DeepAgentConfig(
            name="Async Agent",
            instructions="Async instructions"
        )
        
        # Create agent
        agent = self.factory.create_agent(config, tools=[], async_mode=True)
        
        # Verify
        mock_async_create_deep_agent.assert_called_once()
        assert self.factory.agents["Async Agent"] == mock_agent
        assert agent == mock_agent
    
    def test_get_agent(self):
        """Test getting agent by name"""
        mock_agent = MagicMock()
        self.factory.agents["test_agent"] = mock_agent
        
        # Get existing agent
        agent = self.factory.get_agent("test_agent")
        assert agent == mock_agent
        
        # Get non-existing agent
        agent = self.factory.get_agent("non_existing")
        assert agent is None
    
    def test_list_agents(self):
        """Test listing all agents"""
        self.factory.agents = {
            "agent1": MagicMock(),
            "agent2": MagicMock(),
            "agent3": MagicMock()
        }
        
        agents = self.factory.list_agents()
        assert len(agents) == 3
        assert "agent1" in agents
        assert "agent2" in agents
        assert "agent3" in agents


class TestSpecializedAgents:
    """Test specialized agent creation methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.factory = DeepAgentFactory()
    
    @patch('src.ai_core.deep_agents.create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_create_research_agent(self, mock_get_llm, mock_create_deep_agent):
        """Test research agent creation"""
        # Setup mocks
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent
        
        # Create search tool
        @tool
        def search_tool(query: str) -> str:
            """Search tool"""
            return f"Results for {query}"
        
        # Create research agent
        agent = self.factory.create_research_agent(
            search_tool=search_tool,
            name="Test Research Agent"
        )
        
        # Verify
        mock_create_deep_agent.assert_called_once()
        call_args = mock_create_deep_agent.call_args
        
        # Check that instructions contain research-specific content
        assert "researcher" in call_args.kwargs['instructions'].lower()
        assert "research" in call_args.kwargs['instructions'].lower()
        
        # Check tools
        assert len(call_args.kwargs['tools']) >= 1
        
        # Check agent is stored
        assert self.factory.agents["Test Research Agent"] == mock_agent
    
    @patch('src.ai_core.deep_agents.create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_create_coding_agent(self, mock_get_llm, mock_create_deep_agent):
        """Test coding agent creation"""
        # Setup mocks
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent
        
        # Create coding agent
        agent = self.factory.create_coding_agent(
            name="Test Coding Agent",
            language="python"
        )
        
        # Verify
        mock_create_deep_agent.assert_called_once()
        call_args = mock_create_deep_agent.call_args
        
        # Check that instructions contain coding-specific content
        assert "python" in call_args.kwargs['instructions'].lower()
        assert "developer" in call_args.kwargs['instructions'].lower()
        
        # Check tools (should include analyze_code tool)
        assert len(call_args.kwargs['tools']) >= 1
        
        # Check agent is stored
        assert self.factory.agents["Test Coding Agent"] == mock_agent
    
    @patch('src.ai_core.deep_agents.create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_create_data_analysis_agent(self, mock_get_llm, mock_create_deep_agent):
        """Test data analysis agent creation"""
        # Setup mocks
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent
        
        # Create data analysis agent
        agent = self.factory.create_data_analysis_agent(
            name="Test Analysis Agent"
        )
        
        # Verify
        mock_create_deep_agent.assert_called_once()
        call_args = mock_create_deep_agent.call_args
        
        # Check that instructions contain analysis-specific content
        assert "analyst" in call_args.kwargs['instructions'].lower()
        assert "analyze" in call_args.kwargs['instructions'].lower()
        
        # Check agent is stored
        assert self.factory.agents["Test Analysis Agent"] == mock_agent


class TestConvenienceFunctions:
    """Test module-level convenience functions"""
    
    @patch('src.ai_core.deep_agents.deep_agent_factory')
    def test_create_deep_agent_from_config(self, mock_factory):
        """Test convenience function for creating agent from config"""
        config = DeepAgentConfig(name="Test")
        tools = []
        
        create_deep_agent_from_config(config, tools, async_mode=True)
        
        mock_factory.create_agent.assert_called_once_with(config, tools, True)
    
    @patch('src.ai_core.deep_agents.deep_agent_factory')
    def test_create_research_deep_agent(self, mock_factory):
        """Test convenience function for creating research agent"""
        @tool
        def search_tool(query: str) -> str:
            return "results"
        
        create_research_deep_agent(
            search_tool=search_tool,
            name="Research",
            additional_tools=None,
            async_mode=False
        )
        
        mock_factory.create_research_agent.assert_called_once_with(
            search_tool, "Research", None, False
        )
    
    @patch('src.ai_core.deep_agents.deep_agent_factory')
    def test_create_coding_deep_agent(self, mock_factory):
        """Test convenience function for creating coding agent"""
        create_coding_deep_agent(
            name="Coder",
            language="python",
            project_path=None,
            async_mode=True
        )
        
        mock_factory.create_coding_agent.assert_called_once_with(
            "Coder", "python", None, True
        )


@pytest.mark.asyncio
class TestRunDeepAgent:
    """Test running deep agents"""
    
    async def test_run_deep_agent_basic(self):
        """Test basic agent execution"""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.ainvoke = MagicMock(return_value={"result": "success"})
        
        # Run agent
        messages = [{"role": "user", "content": "Test message"}]
        result = await run_deep_agent(
            agent=mock_agent,
            messages=messages,
            stream=False
        )
        
        # Verify
        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args
        assert call_args[0][0]["messages"] == messages
        assert result == {"result": "success"}
    
    async def test_run_deep_agent_with_files(self):
        """Test agent execution with files"""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.ainvoke = MagicMock(return_value={"result": "success"})
        
        # Run agent with files
        messages = [{"role": "user", "content": "Test message"}]
        files = {"test.txt": "test content"}
        result = await run_deep_agent(
            agent=mock_agent,
            messages=messages,
            files=files,
            stream=False
        )
        
        # Verify
        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args
        assert call_args[0][0]["messages"] == messages
        assert call_args[0][0]["files"] == files


class TestIntegration:
    """Integration tests for deep agents"""
    
    @patch('src.ai_core.deep_agents.create_deep_agent')
    @patch('src.ai_core.deep_agents.get_llm')
    def test_full_agent_lifecycle(self, mock_get_llm, mock_create_deep_agent):
        """Test complete agent lifecycle"""
        # Setup
        mock_model = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent
        
        # Create factory
        factory = DeepAgentFactory()
        
        # Set default model
        factory.set_default_model("gpt-4")
        
        # Create agent
        config = DeepAgentConfig(
            name="Lifecycle Test Agent",
            instructions="Test the full lifecycle",
            enable_file_system=True,
            enable_planning=True
        )
        
        @tool
        def test_tool(input: str) -> str:
            """Test tool"""
            return f"Processed: {input}"
        
        agent = factory.create_agent(config, tools=[test_tool])
        
        # Verify agent was created
        assert agent == mock_agent
        assert factory.get_agent("Lifecycle Test Agent") == mock_agent
        
        # List agents
        agents = factory.list_agents()
        assert "Lifecycle Test Agent" in agents
        
        # Verify creation call
        mock_create_deep_agent.assert_called_once()
        call_args = mock_create_deep_agent.call_args
        
        # Check configuration was applied
        assert call_args.kwargs['instructions'] == "Test the full lifecycle"
        assert "write_file" in call_args.kwargs['builtin_tools']
        assert "write_todos" in call_args.kwargs['builtin_tools']
    
    def test_error_handling(self):
        """Test error handling in agent creation"""
        factory = DeepAgentFactory()
        
        # Test with invalid tool
        config = DeepAgentConfig(name="Error Test")
        invalid_tool = "not_a_tool"  # String instead of callable
        
        with pytest.raises(Exception):
            # This should fail during tool preparation
            factory._prepare_tools([invalid_tool])

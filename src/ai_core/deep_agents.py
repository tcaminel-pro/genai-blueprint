"""
Deep Agents Integration Module

This module provides the core integration layer for deepagents library,
making it easy to create and manage deep agents within the genai-blueprint framework.
It combines the power of deepagents with the existing LangChain/LangGraph infrastructure.

Key Features:
- Factory functions for creating deep agents
- Integration with existing LLM factory
- Tool management and registration
- File system and state management
- Configurable agents with custom subagents
"""

from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from pathlib import Path
import asyncio
from functools import wraps

from deepagents import create_deep_agent, async_create_deep_agent, create_configurable_agent, async_create_configurable_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langchain.tools import tool as langchain_tool
from loguru import logger
from pydantic import BaseModel, Field

from src.ai_core.llm_factory import get_llm


class DeepAgentConfig(BaseModel):
    """Configuration for Deep Agent instances"""
    
    name: str = Field(default="Deep Agent", description="Name of the agent")
    instructions: str = Field(default="", description="System instructions for the agent")
    model: Optional[str] = Field(default=None, description="Model to use (if None, uses default)")
    builtin_tools: Optional[List[str]] = Field(default=None, description="List of builtin tools to enable")
    on_duplicate_tools: str = Field(default="warn", description="How to handle duplicate tools: warn, error, replace, ignore")
    subagents: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom subagents configuration")
    enable_file_system: bool = Field(default=True, description="Enable virtual file system tools")
    enable_planning: bool = Field(default=True, description="Enable planning tool")
    

class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent"""
    
    name: str = Field(description="Name of the subagent")
    description: str = Field(description="Description of the subagent")
    prompt: str = Field(description="System prompt for the subagent")
    tools: Optional[List[str]] = Field(default=None, description="List of tools available to subagent")
    model_settings: Optional[Dict[str, Any]] = Field(default=None, description="Model settings for subagent")


class DeepAgentFactory:
    """Factory class for creating and managing deep agents"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.default_model: Optional[BaseChatModel] = None
        
    def set_default_model(self, model: Optional[Union[str, BaseChatModel]] = None):
        """Set the default model for all agents"""
        if model is None:
            self.default_model = get_llm()
        elif isinstance(model, str):
            self.default_model = get_llm(llm_id=model)
        else:
            self.default_model = model
            
    def _convert_langchain_tool(self, tool: BaseTool) -> Callable:
        """Convert a LangChain tool to a regular Python function for deepagents"""
        
        @wraps(tool.func if hasattr(tool, 'func') else tool._run)
        def wrapper(**kwargs):
            return tool.run(kwargs)
        
        # Preserve the docstring and name
        wrapper.__doc__ = tool.description
        wrapper.__name__ = tool.name
        
        return wrapper
    
    def _prepare_tools(self, tools: List[Union[BaseTool, Callable]]) -> List[Callable]:
        """Prepare tools for use with deepagents"""
        prepared_tools = []
        
        for tool in tools:
            if isinstance(tool, BaseTool):
                prepared_tools.append(self._convert_langchain_tool(tool))
            else:
                prepared_tools.append(tool)
                
        return prepared_tools
    
    def create_agent(
        self,
        config: DeepAgentConfig,
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False
    ) -> Any:
        """
        Create a deep agent with the given configuration.
        
        Args:
            config: DeepAgentConfig object with agent settings
            tools: List of tools (can be LangChain tools or regular functions)
            async_mode: Whether to create an async agent
            
        Returns:
            A configured deep agent instance
        """
        
        # Prepare tools
        prepared_tools = self._prepare_tools(tools or [])
        
        # Get model
        if config.model:
            model = get_llm(model_name=config.model)
        else:
            model = self.default_model or get_llm()
        
        # Prepare subagents if provided
        subagents = None
        if config.subagents:
            subagents = [
                {
                    "name": sa["name"],
                    "description": sa["description"],
                    "prompt": sa["prompt"],
                    "tools": sa.get("tools"),
                    "model_settings": sa.get("model_settings")
                }
                for sa in config.subagents
            ]
        
        # Prepare builtin tools
        builtin_tools = config.builtin_tools
        if builtin_tools is None and config.enable_file_system:
            builtin_tools = ["write_file", "read_file", "ls", "edit_file"]
        if builtin_tools is None and config.enable_planning:
            if builtin_tools is None:
                builtin_tools = []
            builtin_tools.append("write_todos")
        
        # Create the agent
        create_func = async_create_deep_agent if async_mode else create_deep_agent
        
        # Note: on_duplicate_tools is not supported in deepagents API yet
        agent = create_func(
            tools=prepared_tools,
            instructions=config.instructions,
            model=model,
            subagents=subagents,
            builtin_tools=builtin_tools
        )
        
        # Store the agent
        self.agents[config.name] = agent
        logger.info(f"Created deep agent: {config.name}")
        
        return agent
    
    def create_research_agent(
        self,
        search_tool: Union[BaseTool, Callable],
        name: str = "Research Agent",
        additional_tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False
    ) -> Any:
        """
        Create a specialized research agent.
        
        Args:
            search_tool: The main search tool for research
            name: Name of the agent
            additional_tools: Additional tools beyond search
            async_mode: Whether to create an async agent
            
        Returns:
            A configured research agent
        """
        
        research_instructions = """You are an expert researcher. Your job is to conduct thorough research and write polished, well-formatted reports.

IMPORTANT: Format ALL your responses using proper Markdown syntax:
- Use # for main title, ## for major sections, ### for subsections
- Use **bold** for emphasis and key terms
- Use bullet points (- or *) for lists
- Use numbered lists (1. 2. 3.) when order matters
- Use > for important quotes or key insights
- Use `code` for technical terms
- Use --- for section dividers
- Include tables using | syntax when comparing data
- Add line breaks between sections for readability

## Research Process

1. Search for relevant information using the provided search tool
2. Analyze and synthesize the information
3. Write a comprehensive, well-structured report

## Output Format

Your response should follow this structure:

# [Topic Title]

## Overview
[Brief summary]

## Key Findings
- Finding 1
- Finding 2

## Details
[Detailed information organized in sections]

## Sources
- [Source 1](url)
- [Source 2](url)

Be thorough, objective, and always cite your sources."""

        config = DeepAgentConfig(
            name=name,
            instructions=research_instructions,
            enable_file_system=True,
            enable_planning=True
        )
        
        tools = [search_tool]
        if additional_tools:
            tools.extend(additional_tools)
            
        return self.create_agent(config, tools, async_mode)
    
    def create_coding_agent(
        self,
        name: str = "Coding Agent",
        language: str = "python",
        project_path: Optional[Path] = None,
        async_mode: bool = False
    ) -> Any:
        """
        Create a specialized coding agent.
        
        Args:
            name: Name of the agent
            language: Primary programming language
            project_path: Path to the project directory
            async_mode: Whether to create an async agent
            
        Returns:
            A configured coding agent
        """
        
        coding_instructions = f"""You are an expert {language} developer. Your job is to help with coding tasks including:

1. Writing new code
2. Debugging existing code
3. Refactoring and optimization
4. Writing tests
5. Documentation

## Working Process

1. Understand the requirements
2. Plan the implementation using the planning tool
3. Write code using the file system tools
4. Test and debug as needed
5. Document your work

{"Project path: " + str(project_path) if project_path else ""}

Always write clean, maintainable, and well-documented code following best practices."""

        # Create a code analysis tool
        @langchain_tool
        def analyze_code(code: str) -> str:
            """Analyze code for potential issues, complexity, and suggestions"""
            lines = code.split('\n')
            return f"Code analysis: {len(lines)} lines, complexity analysis would go here"
        
        config = DeepAgentConfig(
            name=name,
            instructions=coding_instructions,
            enable_file_system=True,
            enable_planning=True
        )
        
        return self.create_agent(config, [analyze_code], async_mode)
    
    def create_data_analysis_agent(
        self,
        name: str = "Data Analysis Agent",
        additional_tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False
    ) -> Any:
        """
        Create a specialized data analysis agent.
        
        Args:
            name: Name of the agent
            additional_tools: Additional analysis tools
            async_mode: Whether to create an async agent
            
        Returns:
            A configured data analysis agent
        """
        
        analysis_instructions = """You are an expert data analyst. Your job is to analyze data, find insights, and create reports.

## Analysis Process

1. Understand the data and requirements
2. Plan the analysis approach
3. Perform exploratory data analysis
4. Identify patterns and insights
5. Create visualizations if needed
6. Write a comprehensive analysis report

Use the file system to save intermediate results and final reports.
Be thorough in your analysis and clearly communicate findings."""

        config = DeepAgentConfig(
            name=name,
            instructions=analysis_instructions,
            enable_file_system=True,
            enable_planning=True
        )
        
        tools = additional_tools or []
        
        return self.create_agent(config, tools, async_mode)
    
    def create_configurable_deep_agent(
        self,
        agent_config: Dict[str, Any],
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False,
        recursion_limit: int = 1000
    ) -> Any:
        """
        Create a configurable deep agent that can be customized at runtime.
        
        Args:
            agent_config: Dictionary with agent configuration
            tools: List of tools
            async_mode: Whether to create an async agent
            recursion_limit: Maximum recursion depth
            
        Returns:
            A configurable deep agent
        """
        
        prepared_tools = self._prepare_tools(tools or [])
        
        create_func = async_create_configurable_agent if async_mode else create_configurable_agent
        
        agent = create_func(
            agent_config.get('instructions', ''),
            agent_config.get('subagents', []),
            prepared_tools,
            agent_config={"recursion_limit": recursion_limit}
        )
        
        return agent
    
    def get_agent(self, name: str) -> Optional[Any]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all created agents"""
        return list(self.agents.keys())


# Global factory instance
deep_agent_factory = DeepAgentFactory()


# Convenience functions
def create_deep_agent_from_config(
    config: DeepAgentConfig,
    tools: Optional[List[Union[BaseTool, Callable]]] = None,
    async_mode: bool = False
) -> Any:
    """
    Convenience function to create a deep agent.
    
    Args:
        config: DeepAgentConfig object
        tools: List of tools
        async_mode: Whether to create async agent
        
    Returns:
        Configured deep agent
    """
    return deep_agent_factory.create_agent(config, tools, async_mode)


def create_research_deep_agent(
    search_tool: Union[BaseTool, Callable],
    name: str = "Research Agent",
    additional_tools: Optional[List[Union[BaseTool, Callable]]] = None,
    async_mode: bool = False
) -> Any:
    """
    Convenience function to create a research agent.
    
    Args:
        search_tool: Main search tool
        name: Agent name
        additional_tools: Additional tools
        async_mode: Whether to create async agent
        
    Returns:
        Configured research agent
    """
    return deep_agent_factory.create_research_agent(
        search_tool, name, additional_tools, async_mode
    )


def create_coding_deep_agent(
    name: str = "Coding Agent",
    language: str = "python",
    project_path: Optional[Path] = None,
    async_mode: bool = False
) -> Any:
    """
    Convenience function to create a coding agent.
    
    Args:
        name: Agent name
        language: Programming language
        project_path: Project path
        async_mode: Whether to create async agent
        
    Returns:
        Configured coding agent
    """
    return deep_agent_factory.create_coding_agent(
        name, language, project_path, async_mode
    )


async def run_deep_agent(
    agent: Any,
    messages: List[Dict[str, str]],
    files: Optional[Dict[str, str]] = None,
    stream: bool = False
) -> Union[Dict[str, Any], AsyncIterator[Any]]:
    """
    Run a deep agent with messages.
    
    Args:
        agent: The deep agent to run
        messages: List of message dictionaries
        files: Optional file system state
        stream: Whether to stream responses
        
    Returns:
        Agent response (dict if not streaming, async iterator if streaming)
    """
    
    input_data = {"messages": messages}
    if files:
        input_data["files"] = files
    
    if stream:
        # Return the async generator directly
        return agent.astream(input_data)
    else:
        result = await agent.ainvoke(input_data)
        return result

"""
Deep Agent Demo Application

This demo showcases the capabilities of deepagents integration in the genai-blueprint framework.
It provides an interactive interface to work with different types of deep agents.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain.tools import tool
from loguru import logger

from src.ai_core.deep_agents import (
    DeepAgentConfig,
    create_coding_deep_agent,
    create_research_deep_agent,
    deep_agent_factory,
    run_deep_agent,
)
from src.ai_extra.tools_langchain.web_search_tool import basic_web_search

# Page configuration
st.set_page_config(page_title="Deep Agents Demo", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")


class DeepAgentDemo:
    """Interactive demo application for deep agents"""

    def __init__(self):
        """Initialize the demo application"""
        if "agent_type" not in st.session_state:
            st.session_state.agent_type = None
        if "current_agent" not in st.session_state:
            st.session_state.current_agent = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "files" not in st.session_state:
            st.session_state.files = {}

    def render_sidebar(self):
        """Render the sidebar with agent selection and configuration"""
        with st.sidebar:
            st.title("🤖 Deep Agent Configuration")

            # Agent type selection
            agent_type = st.selectbox(
                "Select Agent Type",
                ["Research Agent", "Coding Agent", "Data Analysis Agent", "Custom Agent"],
                key="agent_selector",
            )

            # Agent-specific configuration
            if agent_type == "Research Agent":
                self.render_research_config()
            elif agent_type == "Coding Agent":
                self.render_coding_config()
            elif agent_type == "Data Analysis Agent":
                self.render_analysis_config()
            else:
                self.render_custom_config()

            # Create/Reset agent button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Agent", type="primary", use_container_width=True):
                    self.create_agent(agent_type)
            with col2:
                if st.button("Reset", type="secondary", use_container_width=True):
                    self.reset_agent()

            # Display agent status
            st.divider()
            if st.session_state.current_agent:
                st.success(f"✅ {agent_type} is active")
                st.caption(f"Created at: {st.session_state.get('agent_created_at', 'Unknown')}")
            else:
                st.info("No agent created yet")

            # File system viewer
            if st.session_state.files:
                st.divider()
                st.subheader("📁 Virtual File System")
                for filename, content in st.session_state.files.items():
                    with st.expander(filename):
                        st.code(content[:500] + "..." if len(content) > 500 else content)

    def render_research_config(self):
        """Render research agent configuration"""
        st.subheader("Research Agent Settings")

        st.session_state.research_depth = st.select_slider(
            "Research Depth", options=["quick", "moderate", "comprehensive"], value="moderate"
        )

        st.session_state.use_web_search = st.checkbox("Enable Web Search (requires Tavily API key)", value=False)

        if st.session_state.use_web_search:
            st.session_state.tavily_api_key = st.text_input(
                "Tavily API Key", type="password", help="Get your API key from https://tavily.com"
            )

        st.session_state.focus_areas = st.text_area(
            "Focus Areas (one per line)",
            placeholder="e.g.,\nTechnical details\nPractical applications\nRecent developments",
        )

    def render_coding_config(self):
        """Render coding agent configuration"""
        st.subheader("Coding Agent Settings")

        st.session_state.programming_language = st.selectbox(
            "Programming Language", ["python", "javascript", "typescript", "java", "go", "rust"], index=0
        )

        st.session_state.coding_style = st.multiselect(
            "Coding Preferences",
            ["Type hints", "Comprehensive tests", "Documentation", "Performance optimization"],
            default=["Type hints", "Documentation"],
        )

        st.session_state.use_subagents = st.checkbox(
            "Enable Specialized Sub-agents",
            value=True,
            help="Includes test-writer, code-reviewer, and documentation-writer",
        )

        st.session_state.project_path = st.text_input("Project Path (optional)", placeholder="/path/to/project")

    def render_analysis_config(self):
        """Render data analysis agent configuration"""
        st.subheader("Data Analysis Settings")

        st.session_state.analysis_type = st.selectbox(
            "Analysis Type", ["Exploratory", "Statistical", "Predictive", "Descriptive"]
        )

        st.session_state.visualization = st.checkbox("Generate Visualizations", value=True)

        st.session_state.report_format = st.selectbox("Report Format", ["Markdown", "HTML", "PDF", "JSON"])

    def render_custom_config(self):
        """Render custom agent configuration"""
        st.subheader("Custom Agent Settings")

        st.session_state.custom_name = st.text_input("Agent Name", value="Custom Deep Agent")

        st.session_state.custom_instructions = st.text_area(
            "System Instructions", placeholder="Describe what this agent should do...", height=200
        )

        st.session_state.enable_planning = st.checkbox("Enable Planning Tool", value=True)

        st.session_state.enable_filesystem = st.checkbox("Enable File System", value=True)

        # Custom tools section
        st.subheader("Custom Tools")
        st.caption("Define custom tools as JSON")
        st.session_state.custom_tools_json = st.text_area(
            "Tools Definition (JSON)", value="[]", height=100, help="Array of tool definitions"
        )

    def create_agent(self, agent_type: str):
        """Create the selected agent type"""
        try:
            if agent_type == "Research Agent":
                self.create_research_agent()
            elif agent_type == "Coding Agent":
                self.create_coding_agent()
            elif agent_type == "Data Analysis Agent":
                self.create_analysis_agent()
            else:
                self.create_custom_agent()

            st.session_state.agent_type = agent_type
            st.session_state.agent_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"✅ {agent_type} created successfully!")

        except Exception as e:
            st.error(f"Failed to create agent: {str(e)}")
            logger.error(f"Agent creation error: {e}")

    def create_research_agent(self):
        """Create a research agent"""
        if st.session_state.get("use_web_search") and st.session_state.get("tavily_api_key"):
            from tavily import TavilyClient

            client = TavilyClient(api_key=st.session_state.tavily_api_key)

            @tool
            def web_search(query: str) -> Dict[str, Any]:
                """Search the web using Tavily"""
                return client.search(query, max_results=5)
        else:

            @tool
            def web_search(query: str) -> str:
                return basic_web_search(query)

        # Create the agent
        agent = create_research_deep_agent(search_tool=web_search, name="Interactive Research Agent", async_mode=True)

        st.session_state.current_agent = agent

    def create_coding_agent(self):
        """Create a coding agent"""
        project_path = st.session_state.get("project_path")
        if project_path:
            project_path = Path(project_path)

        agent = create_coding_deep_agent(
            name="Interactive Coding Agent",
            language=st.session_state.get("programming_language", "python"),
            project_path=project_path,
            async_mode=True,
        )

        st.session_state.current_agent = agent

    def create_analysis_agent(self):
        """Create a data analysis agent"""

        @tool
        def analyze_data(data: str) -> str:
            """Analyze data and return insights"""
            return f"Analysis of data: {len(data)} characters processed"

        config = DeepAgentConfig(
            name="Data Analysis Agent",
            instructions="""You are an expert data analyst. Analyze data, find patterns, and create reports.""",
            enable_file_system=True,
            enable_planning=True,
        )

        agent = deep_agent_factory.create_agent(config=config, tools=[analyze_data], async_mode=True)

        st.session_state.current_agent = agent

    def create_custom_agent(self):
        """Create a custom agent"""
        config = DeepAgentConfig(
            name=st.session_state.get("custom_name", "Custom Agent"),
            instructions=st.session_state.get("custom_instructions", ""),
            enable_file_system=st.session_state.get("enable_filesystem", True),
            enable_planning=st.session_state.get("enable_planning", True),
        )

        # Parse custom tools if provided
        tools = []
        try:
            tools_json = st.session_state.get("custom_tools_json", "[]")
            if tools_json and tools_json != "[]":
                # This is simplified - in production, properly parse and create tools
                pass
        except Exception as e:
            st.warning(f"Could not parse custom tools: {e}")

        agent = deep_agent_factory.create_agent(config=config, tools=tools, async_mode=True)

        st.session_state.current_agent = agent

    def reset_agent(self):
        """Reset the current agent and clear history"""
        st.session_state.current_agent = None
        st.session_state.agent_type = None
        st.session_state.chat_history = []
        st.session_state.files = {}
        st.info("Agent reset successfully")

    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🤖 Deep Agent Interactive Demo")

        if not st.session_state.current_agent:
            st.info("👈 Please create an agent from the sidebar to get started")

            # Show example use cases
            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔬 Research Agent")
                st.write(
                    "Conduct comprehensive research on any topic with web search, note-taking, and report generation."
                )
                if st.button("Try Research Agent", use_container_width=True):
                    st.session_state.agent_selector = "Research Agent"
                    st.rerun()

            with col2:
                st.subheader("💻 Coding Agent")
                st.write("Write, debug, refactor, and test code with specialized sub-agents for different tasks.")
                if st.button("Try Coding Agent", use_container_width=True):
                    st.session_state.agent_selector = "Coding Agent"
                    st.rerun()

            with col3:
                st.subheader("📊 Analysis Agent")
                st.write("Analyze data, find insights, and create comprehensive reports with visualizations.")
                if st.button("Try Analysis Agent", use_container_width=True):
                    st.session_state.agent_selector = "Data Analysis Agent"
                    st.rerun()

            return

        # Chat history display
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask your agent anything..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    response = asyncio.run(self.get_agent_response(prompt))
                    st.write(response)

                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    async def get_agent_response(self, prompt: str) -> str:
        """Get response from the current agent"""
        try:
            messages = [{"role": "user", "content": prompt}]

            # Include files if they exist
            files = st.session_state.files if st.session_state.files else None

            result = await run_deep_agent(
                agent=st.session_state.current_agent, messages=messages, files=files, stream=False
            )

            # Update files if changed
            if "files" in result:
                st.session_state.files = result["files"]

            # Extract response
            if "messages" in result and result["messages"]:
                return result["messages"][-1].content
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error getting agent response: {e}")
            return f"Error: {str(e)}"

    def render_examples(self):
        """Render example prompts based on agent type"""
        if st.session_state.current_agent and st.session_state.agent_type:
            st.divider()
            st.subheader("💡 Example Prompts")

            examples = self.get_example_prompts(st.session_state.agent_type)

            cols = st.columns(3)
            for i, example in enumerate(examples):
                with cols[i % 3]:
                    if st.button(example, use_container_width=True, key=f"example_{i}"):
                        st.session_state.chat_history.append({"role": "user", "content": example})
                        st.rerun()

    def get_example_prompts(self, agent_type: str) -> List[str]:
        """Get example prompts for the agent type"""
        examples = {
            "Research Agent": [
                "Research the latest developments in quantum computing",
                "What are the best practices for building LLM applications?",
                "Compare different vector databases for AI applications",
            ],
            "Coding Agent": [
                "Write a function to validate email addresses",
                "Debug this code: def avg(nums): return sum(nums)/len(nums)",
                "Refactor this function to be more Pythonic",
            ],
            "Data Analysis Agent": [
                "Analyze this dataset and find key insights",
                "Create a statistical summary of the data",
                "Generate a report on data quality issues",
            ],
            "Custom Agent": ["Help me with my task", "What can you do?", "Show me your capabilities"],
        }

        return examples.get(agent_type, examples["Custom Agent"])

    def run(self):
        """Run the demo application"""
        # Render sidebar
        self.render_sidebar()

        # Render main chat interface
        self.render_chat_interface()

        # Render examples if agent is active
        self.render_examples()

        # Footer
        st.divider()
        st.caption("Built with DeepAgents and GenAI Blueprint | Powered by LangChain & LangGraph")


def main():
    """Main entry point for the demo"""
    demo = DeepAgentDemo()
    demo.run()


if __name__ == "__main__":
    main()

"""Configuration page for GenAI Lab.

Displays current configuration information and provides controls for:
- LLM model selection
- Cache settings
- Debug/verbose modes
- Monitoring configuration
"""

import os

import streamlit as st
from langchain.globals import set_debug, set_verbose

from src.ai_core.cache import LlmCache
from src.ai_core.llm import PROVIDER_INFO, LlmFactory
from src.utils.config_mngr import global_config


def display_config_info() -> None:
    """Display current configuration and available API keys."""
    config = global_config()

    st.subheader("Current Configuration")
    st.info(f"**Selected configuration:** {config.selected_config}")

    # Display key configuration values
    st.markdown("**Key Configuration Values:**")
    config_data = []

    # LLM configuration
    try:
        llm_model = config.get_str("llm.default_model", "Not set")
        config_data.append({"Setting": "LLM Default Model", "Value": llm_model})
    except Exception:
        config_data.append({"Setting": "LLM Default Model", "Value": "Not available"})

    # Embeddings configuration
    try:
        embeddings_model = config.get_str("embeddings.default_model", "Not set")
        config_data.append({"Setting": "Embeddings Default Model", "Value": embeddings_model})
    except Exception:
        config_data.append({"Setting": "Embeddings Default Model", "Value": "Not available"})

    # Vector store configuration
    try:
        vector_store = config.get_str("vector_store.default", "Not set")
        config_data.append({"Setting": "Vector Store Default", "Value": vector_store})
    except Exception:
        config_data.append({"Setting": "Vector Store Default", "Value": "Not available"})

    # Cache configuration
    try:
        cache_method = config.get_str("llm.cache", "Not set")
        config_data.append({"Setting": "LLM Cache Method", "Value": cache_method})
    except Exception:
        config_data.append({"Setting": "LLM Cache Method", "Value": "Not available"})

    # Monitoring configuration
    try:
        monitoring = config.get_str("monitoring.default", "Not set")
        config_data.append({"Setting": "Monitoring Default", "Value": monitoring})
    except Exception:
        config_data.append({"Setting": "Monitoring Default", "Value": "Not available"})

    if config_data:
        st.table(config_data)

    st.subheader("Available LLM API Keys")

    # Create a table showing API key status
    api_key_data = []
    for provider, (_, key_name) in PROVIDER_INFO.items():
        if key_name and key_name in os.environ:
            is_set = key_name in os.environ
            status = "✅ Set" if is_set else "❌ Not set"
            api_key_data.append({"Provider": provider, "Environment Variable": key_name, "Status": status})

    if api_key_data:
        st.table(api_key_data)
    else:
        st.warning("No API key information available")


def llm_configuration_section() -> None:
    """LLM configuration controls."""
    st.subheader("LLM Configuration")

    # LLM Model Selection
    current_llm = global_config().get_str("llm.default_model")
    available_models = LlmFactory().known_items()

    try:
        index = available_models.index(current_llm)
    except ValueError:
        index = 0

    selected_llm = st.selectbox(
        "Default LLM Model",
        available_models,
        index=index,
        key="select_llm",
        help="Select the default LLM model to use across the application",
    )

    if selected_llm != current_llm:
        global_config().set("llm.default_model", str(selected_llm))
        st.success(f"Default LLM changed to: {selected_llm}")

    # Cache Configuration
    cache_method = st.selectbox(
        "Cache Method",
        ["memory", "sqlite", "no_cache"],
        index=1,  # Default to sqlite
        help="Choose caching strategy for LLM responses",
    )
    LlmCache.set_method(cache_method)

    # Debug and Verbose Settings
    col1, col2 = st.columns(2)

    with col1:
        debug_mode = st.checkbox("Debug Mode", value=False, help="Enable LangChain debug mode for detailed logging")
        set_debug(debug_mode)

    with col2:
        verbose_mode = st.checkbox(
            "Verbose Mode", value=True, help="Enable LangChain verbose mode for operation details"
        )
        set_verbose(verbose_mode)


def mcp_servers_section() -> None:
    """Display information about available MCP servers and their tools."""
    st.subheader("MCP Servers & Tools")

    import asyncio

    from src.ai_core.mcp_client import get_mcp_tools_info

    async def display_tools():
        with st.spinner("Loading MCP servers and tools..."):
            tools_info = await get_mcp_tools_info()
            if not tools_info:
                st.info("No MCP servers found.")
                return

        for server_name, tools in tools_info.items():
            with st.expander(f"Server: {server_name}", expanded=False):
                # Convert tools dict to list of dicts for dataframe display
                from devtools import debug

                debug(tools.items())
                table_data = [{"Tool": tool, "Description": desc} for tool, desc in tools.items()]
                st.dataframe(
                    table_data,
                    column_config={
                        "Tool": st.column_config.Column(width="small"),
                        "Description": st.column_config.TextColumn(
                            width="large",
                            help="Description of the tool",
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

    asyncio.run(display_tools())


def monitoring_configuration_section() -> None:
    """Monitoring configuration controls."""
    st.subheader("Monitoring Configuration")

    monitoring_options = []

    # Check for Lunary
    if "LUNARY_APP_ID" in os.environ:
        lunary_enabled = st.checkbox(
            "Use Lunary.ai for monitoring", value=False, disabled=True, help="Lunary monitoring (currently disabled)"
        )
        if lunary_enabled:
            global_config().set("monitoring.lunary", "true")
        monitoring_options.append("Lunary.ai detected")

    # Check for LangSmith
    if "LANGCHAIN_API_KEY" in os.environ:
        langsmith_enabled = st.checkbox(
            "Use LangSmith for monitoring", value=True, help="Enable LangSmith tracing and monitoring"
        )

        if langsmith_enabled:
            global_config().set("monitoring.langsmith", "true")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = global_config().get_str("monitoring.project")
            os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "1.0"
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            global_config().set("monitoring.langsmith", "false")

        monitoring_options.append("LangSmith detected")

    if not monitoring_options:
        st.info(
            "No monitoring services detected. Set LUNARY_APP_ID or LANGCHAIN_API_KEY environment variables to enable monitoring."
        )


def main() -> None:
    """Main configuration page."""
    st.title("⚙️ Configuration")
    st.markdown("Configure your GenAI Lab settings and view current configuration status.")

    # Configuration Information Section
    with st.expander("📋 Configuration Information", expanded=False):
        display_config_info()

    # LLM Configuration Section
    with st.expander("🤖 LLM Settings", expanded=True):
        llm_configuration_section()

        # LLM Test Section
        st.subheader("Test LLM")
        test_input = st.text_area(
            "Test Input", value="Tell me a joke in french", help="Enter text to test the currently configured LLM"
        )

        if st.button("Run Test"):
            from langchain_core.messages import HumanMessage

            from src.ai_core.llm import get_llm

            with st.spinner("Running LLM test..."):
                try:
                    llm = get_llm()
                    response = llm.invoke([HumanMessage(content=test_input)])
                    st.success("LLM Response:")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Error running LLM test: {str(e)}")

    # MCP Servers Section
    with st.expander("🛠️ MCP Servers & Tools", expanded=False):
        mcp_servers_section()

    # Monitoring Configuration Section
    with st.expander("📊 Monitoring Settings", expanded=False):
        monitoring_configuration_section()

    # Additional Information
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Cache Method**: Use 'sqlite' for persistent caching across sessions, 'memory' for session-only caching
    - **Debug Mode**: Enable for troubleshooting LangChain operations
    - **Monitoring**: LangSmith provides detailed tracing of LLM calls and chain executions
    """)


if __name__ == "__main__":
    main()
else:
    main()

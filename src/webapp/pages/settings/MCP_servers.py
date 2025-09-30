"""MCP Configuration page for GenAI Lab."""

import streamlit as st


def mcp_servers_section() -> None:
    """Display information about available MCP servers and their tools."""
    st.subheader("MCP Servers & Tools")

    import asyncio

    from genai_tk.core.mcp_client import get_mcp_tools_info

    async def display_tools():
        with st.spinner("Loading MCP servers and tools..."):
            tools_info = await get_mcp_tools_info()
            if not tools_info:
                st.info("No MCP servers found.")
                return

        for server_name, tools in tools_info.items():
            with st.expander(f"Server: {server_name}", expanded=False):
                # Convert tools dict to list of dicts for dataframe display

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
                    width="stretch",
                )

    asyncio.run(display_tools())


def main() -> None:
    """Main configuration page."""
    st.title("MCP Configuration")
    st.markdown("See Known MCP Server.")

    mcp_servers_section()


if __name__ == "__main__":
    main()
else:
    main()

#!/usr/bin/env python
"""
Test script for DeepAgents integration
This demonstrates the deep agents without requiring external API keys
"""

import asyncio

from genai_tk.ai_core.deep_agents import DeepAgentConfig, deep_agent_factory, run_deep_agent
from langchain.tools import tool


# Create a simple mock search tool
@tool
def mock_search(query: str) -> str:
    """Mock search tool for demonstration"""
    return f"""
    Search Results for: {query}
    
    1. Recent breakthrough in quantum error correction using topological codes
       - Researchers achieved 99.9% fidelity in quantum operations
       - New approach uses machine learning for error prediction
    
    2. IBM announces 1000-qubit quantum processor roadmap
       - Expected release by 2025
       - Focus on practical quantum advantage applications
    
    3. Google's quantum supremacy challenged by classical algorithms
       - New classical simulation methods show competitive performance
       - Debate continues on defining quantum advantage
    """


async def main():
    print("🤖 Deep Agent Demo - Research Agent")
    print("=" * 50)

    # Set up to use the fake model that doesn't require API keys
    deep_agent_factory.set_default_model("parrot_local_fake")

    # Create a research agent
    print("\n📚 Creating Research Agent...")
    agent = deep_agent_factory.create_research_agent(
        search_tool=mock_search, name="Demo Research Agent", async_mode=True
    )

    # Prepare the query
    query = "Latest developments in quantum computing"
    print(f"\n🔍 Researching: {query}")
    print("-" * 50)

    # Run the agent
    messages = [{"role": "user", "content": f"Research this topic: {query}"}]

    try:
        result = await run_deep_agent(agent=agent, messages=messages, stream=False)

        # Display results
        if "messages" in result and result["messages"]:
            print("\n📊 Research Results:")
            print("-" * 50)
            print(result["messages"][-1].content)

        # Show any files created
        if "files" in result and result["files"]:
            print("\n📁 Files Created:")
            print("-" * 50)
            for filename, content in result["files"].items():
                print(f"\n📄 {filename}:")
                print(content[:500] + "..." if len(content) > 500 else content)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("✅ Demo Complete!")


if __name__ == "__main__":
    # Note: This uses a mock LLM that doesn't require API keys
    # You can set BLUEPRINT_CONFIG environment variable to use different models

    # For this demo, we'll use a simple configuration
    import os

    # Use a configuration that doesn't require external APIs
    if "BLUEPRINT_CONFIG" not in os.environ:
        os.environ["BLUEPRINT_CONFIG"] = "pytest"  # This uses simpler defaults

    asyncio.run(main())

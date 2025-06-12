import asyncio
import html
import streamlit as st
from browser_use import Agent
from src.ai_core.llm import get_llm

# Initialize LLM
llm = get_llm(llm_id="gpt_4o_azure")

# Configure Streamlit page
#st.set_page_config(page_title="Browser Agent Controller", layout="wide")
st.title("Web Navigation Agent Control Panel")


def main():
    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "page_content" not in st.session_state:
        st.session_state.page_content = ""

    # Control sidebar
    with st.sidebar:
        st.header("Agent Controls")
        url = st.text_input("Enter target URL:", key="url")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.running):
                if url:
                    st.session_state.agent = Agent(task=url, llm=llm)
                    st.session_state.running = True
                    st.session_state.page_content = ""
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.running):
                st.session_state.running = False
                st.session_state.agent = None

    # Status display
    with st.container():
        (status_col,) = st.columns(1)
        with status_col:
            if st.session_state.running:
                st.success("✅ Agent is actively browsing")
                if st.session_state.agent:
                    # Run agent and get page content
                    try:
                        st.session_state.page_content = asyncio.run(st.session_state.agent.run())
                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")
                        st.session_state.running = False
            else:
                st.info("⏸️ Agent is stopped")

    # Display content in iframe
    if st.session_state.page_content:
        st.markdown("### Current Page View")
        
        # Escape HTML content and wrap in proper document structure
        escaped_content = html.escape(st.session_state.page_content)
        full_html = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ margin: 0; padding: 1rem; }}
                </style>
            </head>
            <body>
                {escaped_content}
            </body>
        </html>
        """
        
        # Display in iframe with proper attributes
        st.markdown(
            f'<iframe srcdoc="{full_html}" '
            'style="width:100%; height:600px; border:1px solid #ddd; '
            'border-radius: 8px; margin: 1rem 0;"></iframe>',
            unsafe_allow_html=True
        )

    # Debug section
    with st.expander("Raw Content Debug"):
        if st.session_state.page_content:
            st.code(st.session_state.page_content[:2000] + "...")  # Show first 2000 chars


if __name__ == "__main__":
    main()

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from src.ai_core.llm import get_llm


def load_web_pages(urls: list[str]) -> str:
    """Load content from the given web page URLs using LangChain's WebBaseLoader."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


def init_session() -> None:
    """Initialize session state for chat history and context."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "context_loaded" not in st.session_state:
        # Load web pages content and add as system prompt context.
        web_content = load_web_pages(WEB_PAGES)
        st.session_state.context = web_content
        st.session_state.chat_history.append({"role": "system", "message": f"Web context:\n{web_content}"})
        st.session_state.context_loaded = True


def display_messages() -> None:
    """Display the messages stored in session state."""
    for message in st.session_state.chat_history:
        role = message["role"]
        msg = message["message"]
        st.markdown(f"**{role.capitalize()}**: {msg}")


def main() -> None:
    st.title("Chat with LLM")
    init_session()
    display_messages()

    user_message: str = st.text_input("Your message", key="user_input")
    if st.button("Send") and user_message.strip() != "":
        # Append user message.
        st.session_state.chat_history.append({"role": "user", "message": user_message.strip()})
        # Combine the full conversation context.
        full_context = "\n".join(f"{m['role']}: {m['message']}" for m in st.session_state.chat_history)
        llm = get_llm()
        # Assuming the LLM instance is callable and returns a response.
        response = llm(full_context)
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        st.rerun()


if __name__ == "__main__":
    main()

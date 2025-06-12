# Import necessary libraries
import streamlit as st
from browser_use import Agent

from src.ai_core.llm import get_llm

llm = get_llm(llm_id="gpt_4o_azure")



# Streamlit application
def main():
    st.title("Browser Use Agent with Streamlit")

    # Input for URL
    query = st.text_input("query:")
    if query:
        agent = Agent(task=query, llm=llm)
        await agent.run()

...
    st.html(page_content, height=600)


main()

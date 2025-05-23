"""LLM Augmented Autonomous Agent for Maintenance.

This module provides tools and agents for assisting maintenance engineers with various tasks
including procedure retrieval, planning information, sensor data, and ERP system queries.

The main components are:
- Maintenance procedure vector stores for similarity search
- SQL query tools for maintenance planning data
- Sensor data retrieval tools
- ERP system query tools
- A maintenance agent that orchestrates these tools
"""

#
# Complete doc and docstring AI!
from datetime import datetime
from functools import cache
from pathlib import Path
from textwrap import dedent

import streamlit as st
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, tool
from langchain.vectorstores.base import VectorStore
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from loguru import logger

from src.ai_core.embeddings import EmbeddingsFactory
from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, def_prompt
from src.ai_core.vector_store import VectorStoreFactory
from src.ai_extra.sql_agent import create_sql_querying_graph
from src.demos.maintenance_agent.maintenance_data import dummy_database
from src.utils.config_mngr import global_config

# Tools setup
PROCEDURES = [
    "procedure_turbine.txt",
    "procedure_generator.txt",
    "procedure_cooling_system.txt",
]

DATA_PATH = Path(global_config().get_str("documents.base")) / "maintenance"

VECTOR_STORE_ID = "InMemory"


@st.cache_resource(show_spinner=True)
def maintenance_procedure_vectors(text: str) -> VectorStore:
    """Index and store a document for similarity matching using embeddings.

    Args:
        text: Name of the text file containing maintenance procedures

    Returns:
        VectorStore: Configured vector store with indexed documents
    """
    vs_factory = VectorStoreFactory(
        id=VECTOR_STORE_ID,
        collection_name="maintenance_procedure",
        embeddings_factory=EmbeddingsFactory(),
    )

    loader = TextLoader(str(DATA_PATH / text))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vs_factory.add_documents(texts)
    return vs_factory.vector_store


examples = [
    {
        "input": "Tasks assigned to employee 'employee_name' between '2023-10-22' and '2023-10-28'.",
        "query": """SELECT task FROM "tasks" WHERE employee = 'employee_name' AND start_date > '2023-10-22' and end_date  < '2023-10-28';""",
    },
    {
        "input": "List of procedures that employee 'employee_name' knows. ",
        "query": "SELECT  DISTINCT employee, procedure FROM tasks WHERE employee = 'employee_name';",
    },
]


@cache
def create_maintenance_tools() -> list[BaseTool]:
    """Create and configure the tools for the Maintenance Agent.

    Returns:
        list[BaseTool]: List of configured tools including:
            - Planning info retrieval
            - Current time
            - Maintenance times
            - Maintenance issues
            - Procedure retrieval
            - ERP system queries
            - Sensor data
    """
    logger.info("create tools")

    @tool
    def get_planning_info(query: str) -> str:
        """Useful for when you need to answer questions about tasks assigned to employees."""

        db = SQLDatabase.from_uri(dummy_database())
        graph = create_sql_querying_graph(get_llm(), db, examples=examples[:5])
        result = graph.invoke({"question": query})
        return result["answer"]

    @tool
    def get_current_time() -> str:
        """A tool to get the current time."""
        return datetime.now().isoformat()

    @tool
    def get_maintenance_times(area: str) -> str:
        """Useful to searches the list of previous maintenance time for the given area."""
        result = """
        -29/05/2018  13:45
        -16/05/2019  10:00
        -22/07/2020  9:45
        """
        return dedent_ws(result)

    @tool
    def get_maintenance_issues(area: str, time: str) -> str:
        """Searches for the issues during the maintenance of a given area at a given time."""
        result = f"""
        Big big problem in area {area} at {time}
        """
        return dedent(result)

    @tool
    def maintenance_procedure_retriever(full_query: str) -> str:
        """Answer to any questions about maintenance procedures, such as tasks, prerequisite, spare parts, required tools etc."""
        system_prompt = (
            "Use the given context to answer the question. If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. \n"
            "Context: {context}"
        )
        prompt = def_prompt(system_prompt, "{input}")
        retriever = maintenance_procedure_vectors(PROCEDURES[0]).as_retriever()
        llm = get_llm()
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain.invoke({"input": full_query})

    @tool
    def get_info_from_erp(tools_name: list[str], time: str) -> str:
        """Useful to answer questions about tools, such as availability, localization, ..."""
        result = ""
        for too_name in tools_name:
            availability = too_name[0].lower() < "r"
            hash = str(too_name.__hash__())
            localization = f"R{hash[1:3]}L{hash[3:7]}"
            result += f"{too_name}:\n\t- availability: {availability} \n\t- localization:{localization}\n"
        return result

    @tool
    def get_sensor_values(sensor_name: list[str], start_time: str, end_time: str) -> str:
        """Useful to know the values of a given sensor during a time range."""
        print(sensor_name, start_time, end_time)
        ls = ",".join([f"'{n}'" for n in sensor_name])
        db = SQLDatabase.from_uri(dummy_database())
        sql = f"""SELECT date, sensor, value, unit FROM  sensor_data
                  WHERE  sensor IN ({ls}) and date >= '{start_time}' and date <= '{end_time}'"""
        print(sql)
        result = db.run(sql)
        return str(result)

    return [
        get_maintenance_times,
        get_maintenance_issues,
        get_planning_info,
        get_current_time,
        maintenance_procedure_retriever,
        get_info_from_erp,
        get_sensor_values,
    ]


def create_maintenance_agent(
    verbose: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
    metadata: dict | None = None,
) -> Runnable:
    """Create a maintenance agent that returns a runnable AgentExecutor.

    Args:
        verbose: Whether to enable verbose logging
        callbacks: Optional list of callback handlers
        metadata: Additional metadata for the agent

    Returns:
        Runnable: Configured agent executor ready to handle maintenance queries
    """
    # Get the LLM info
    # Create tools
    tools = create_maintenance_tools()
    system = dedent_ws(
        """
        You are a helpful assistant to help a maintenance engineer.
        To do so, you have different tools (functions)  to access maintenance planning, spares etc.
        Make sure to use only the provided tools (functions) to answer the user request.
        If you don't find relevant tool, answer "I don't know"
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(get_llm(), tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        metadata={"agentName": "MaintenanceAgent1"} | (metadata or {}),
        callbacks=callbacks,
    )
    # return agent_executor | StrOutputParser()
    return agent_executor

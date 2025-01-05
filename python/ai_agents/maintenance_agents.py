"""
LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""

from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple

import pandas as pd
import streamlit as st
from devtools import debug
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.base import VectorStore
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from loguru import logger

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import LlmFactory
from python.ai_core.prompts import dedent_ws, def_prompt
from python.ai_core.vector_store import VectorStoreFactory
from python.config import get_config_str
from python.demos.maintenance_agent.maintenance_data import dummy_database

# [Rest of the imports and previous code remain the same]

def run_maintenance_agent(
    llm_factory: LlmFactory,
    verbose: bool = False,
    extra_callbacks: list[BaseCallbackHandler] | None = None,
    extra_metadata: dict | None = None,
) -> Runnable:
    """Create a runnable maintenance agent that can be invoked with a query."""
    agent_executor = create_maintenance_agent(
        llm_factory, 
        verbose, 
        extra_callbacks, 
        extra_metadata
    )

    def process_query(query: str):
        cfg = RunnableConfig()
        cfg["callbacks"] = extra_callbacks or []
        cfg["metadata"] = extra_metadata or {}
        
        result = agent_executor.invoke({"input": query}, cfg)
        return result["output"]

    return RunnablePassthrough().bind(func=process_query)

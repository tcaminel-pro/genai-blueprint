"""LLM Augmented Autonomous Agent for Maintenance.

Provides tools and agents for maintenance-related tasks, including:
- Retrieving maintenance procedures
- Querying planning information
- Accessing sensor data
- Managing maintenance-related inquiries
"""

from datetime import datetime
from functools import cache
from pathlib import Path
from textwrap import dedent

import streamlit as st
from devtools import debug
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

# Rest of the code remains the same...

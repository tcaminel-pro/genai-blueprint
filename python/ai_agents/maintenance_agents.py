"""
LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""

from datetime import datetime
from functools import cached_property
from textwrap import dedent
from typing import Literal, Tuple, Union

import pandas as pd
import streamlit as st
from devtools import debug
from langchain.agents import (
    AgentExecutor,
    AgentType,
    create_sql_agent,
    create_tool_calling_agent,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.base import VectorStore
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities.sql_database import SQLDatabase, truncate_word
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from loguru import logger

# Pydantic v1 required - https://python.langchain.com/v0.1/docs/guides/development/pydantic_compatibility/
from pydantic.v1 import BaseModel, Field

# from pydantic import BaseModel, Field
from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import LlmFactory
from python.ai_core.prompts import def_prompt
from python.ai_core.vector_store import VectorStoreFactory
from python.demos.maintenance_agent.maintenance_data import DATA_PATH, dummy_database

# Tools setup
PROCEDURES = [
    "procedure_turbine.txt",
    "procedure_generator.txt",
    "procedure_cooling_system.txt",
]


class SQLDatabaseExt(SQLDatabase):
    sql: str | None = None
    result: str | None = None

    def engine(self):
        return self._engine

    def dataframe(self, command: str):
        with self._engine.connect() as connection:
            connection.exec_driver_sql("SET search_path TO %s", (self._schema,))
            return pd.read_sql(command, connection)

    def run(
        self,
        command: str,
        fetch: Union[Literal["all"], Literal["one"]] = "all",
    ) -> str:
        """ """
        result = self._execute(command, fetch)
        res = [
            tuple(truncate_word(c, length=self._max_string_length) for c in r.values())
            for r in result
        ]
        setattr(res, "sql", command)
        if not res:
            return ""
        else:
            return str(res)


@st.cache_resource(show_spinner=True)
def maintenance_procedure_vectors(text: str) -> VectorStore:
    """Index and store a document for similarity matching (through embeddings).

    Embeddings are stored in a vector-store.
    """

    vs_factory = VectorStoreFactory(
        id="Chroma_in_memory",
        collection_name="maintenance_procedure",
        embeddings_factory=EmbeddingsFactory(),
    )

    loader = TextLoader(str(DATA_PATH / text))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vs_factory.add_documents(texts)
    return vs_factory.vector_store


class MaintenanceAgent(BaseModel):
    """Maintenance Agent class"""

    llm_factory: LlmFactory
    embeddings_factory: EmbeddingsFactory
    tools: list[BaseTool] = Field(default_factory=list)
    _agent_executor: AgentExecutor | None = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        ignored_types = (cached_property,)

    def create_sql_agent(self) -> AgentExecutor:
        """Create an agent for Text-2-SQL.

        Inspired by: https://python.langchain.com/docs/use_cases/qa_structured/sql
        """
        # fmt: off
        few_shots = {
            "Tasks assigned to employee 'employee_name' between '2023-10-22' and '2023-10-28'.": 
                """SELECT task FROM "tasks" WHERE employee = 'employee_name' AND start_date > '2023-10-22' and end_date  < '2023-10-28';""",
            "List of procedures that employee 'employee_name'' knows. ": 
                """SELECT  DISTINCT employee, procedure FROM tasks WHERE employee = 'employee_name';"""
        }

        # fmt: on

        few_shot_docs = [
            Document(page_content=question, metadata={"sql_query": few_shots[question]})
            for question in few_shots.keys()
        ]

        vector_db = VectorStoreFactory(
            id="Chroma_in_memory",
            collection_name="test_maintenance",
            embeddings_factory=self.embeddings_factory,
        ).vector_store
        vector_db.add_documents(few_shot_docs)

        tool_description = """
            This tool will help you understand similar examples to adapt them to the user question.
            Input to this tool should be the user question.
            """
        retriever_tool = create_retriever_tool(
            vector_db.as_retriever(search_kwargs={"k": 1}),
            name="sql_get_similar_examples",
            description=dedent(tool_description),
        )

        custom_suffix = """
            If the query involve time I should get the current time.
            I should first get the similar examples I know. 
            If an example is enough to construct the query, then I build it by adapting values according to actual requests and I execute it.
            Otherwise, I look at the tables in the database to see what I can query, and I should query the schema of the most relevant tables.
            I execute the SQL query to  to retrieve the information.
            """  # noqa: F841

        db = SQLDatabase.from_uri(dummy_database())

        from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX

        prefix = (
            SQL_PREFIX
            + f"\nCurrent date is: {datetime.now().isoformat()}. DO NOT use any SQL date functions like NOW(), DATE(), DATETIME()"
        )

        agent_type = AgentType.OPENAI_FUNCTIONS  # To IMPROVE

        llm = self.llm_factory.get()
        return create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            agent_type=agent_type,
            prefix=prefix,
            verbose=False,
            extra_tools=[retriever_tool],
            # suffix=dedent(custom_suffix),
            max_iterations=5,
        )

    def create_tools(self):
        """Create the tools for our Agent"""

        logger.info("create tools")
        query_database = Tool(
            name="query_maintenance_database",
            func=self.create_sql_agent().run,
            description=(
                "Useful for when you need  to answer questions about tasks assigned to employees."
                "Input should be in the form of a question containing full context"
            ),
        )

        @tool
        def get_current_time() -> str:
            """A tool to get the current time"""
            return datetime.now().isoformat()

        @tool
        def get_maintenance_times(area: str) -> str:
            """Useful to searches the list of previous maintenance time for the given area.
            Returns a list of date and hours"""
            result = """
            -29/05/2018  13:45
            -16/05/2019  10:00
            -22/07/2020  9:45
            """
            return dedent(result)

        @tool
        def get_maintenance_issues(area: str, time: str) -> str:
            """Searches for the issues during the maintenance of a given area at a given time"""
            result = f"""
            Big big problem in area {area} at {time}
            """
            return dedent(result)

        @tool
        def maintenance_procedure_retriever(full_query: str) -> str:
            """
            Answer to any questions about maintenance procedures, such as tasks, prerequisite, spare parts, required tools etc.
            Input should contain full context.
            The questions can ask for information about several tasks, spare parts, tools etc.
            """

            system_prompt = (
                "Use the given context to answer the question. If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. \n"
                "Context: {context}"
            )
            # See https://www.linkedin.com/pulse/beginners-guide-retrieval-chain-using-langchain-vijaykumar-kartha-kuinc/
            prompt = def_prompt(system_prompt, "{input}")
            retriever = maintenance_procedure_vectors(PROCEDURES[0]).as_retriever()
            question_answer_chain = create_stuff_documents_chain(
                self.llm_factory.get(), prompt
            )
            chain = create_retrieval_chain(retriever, question_answer_chain)
            return chain.invoke({"input": full_query})

        # https://vijaykumarkartha.medium.com/beginners-guide-to-creating-ai-agents-with-langchain-eaa5c10973e6

        retriever_tool = create_retriever_tool(
            retriever=maintenance_procedure_vectors(PROCEDURES[0]).as_retriever(),
            name="bla",
            description=dedent(
                """
                Answer to any questions about maintenance procedures, such as tasks, prerequisite, spare parts, required tools etc.
                Input should contain full context.
                The questions can ask for information about several tasks, spare parts, tools etc.
                """
            ),
        )

        @tool
        def get_info_from_erp(tools_name: list[str], time: str) -> str:
            """Useful to answer questions about tools, such as availability, localization, ...
            Input should be a list of tools for which such information are requested"""

            # Mock function . A real function would call an ERP or PLM API
            result = ""
            for too_name in tools_name:
                availability = too_name[0].lower() < "r"
                hash = str(too_name.__hash__())
                localization = f"R{hash[1:3]}L{hash[3:7]}"
                result += f"{too_name}:\n\t- availability: {availability} \n\t- localization:{localization}\n"
            return result

        @tool
        def get_sensor_values(
            sensor_name: list[str], start_time: str, end_time: str
        ) -> str:
            """Useful to know the values of a given sensor during a time range."""
            debug(sensor_name, start_time, end_time)
            ls = ",".join([f"'{n}'" for n in sensor_name])
            db = SQLDatabase.from_uri(dummy_database())
            sql = f"""SELECT date, sensor, value, unit FROM  sensor_data 
                      WHERE  sensor IN ({ls}) and date >= '{start_time}' and date <= '{end_time}'"""
            debug(sql)
            result = db.run(sql)
            return str(result)  # #TODO : Correct and  test

        # https://python.langchain.com/docs/modules/agents/agent_types/structured_chat

        tools = [
            get_maintenance_times,
            get_maintenance_issues,
            query_database,
            get_current_time,
            maintenance_procedure_retriever,
            get_info_from_erp,
            get_sensor_values,
        ]
        self.tools = tools

    def add_tools(self, new_tools: list[BaseTool]):
        tool_descriptions = [tool.description for tool in self.tools]
        self.tools += [
            tool for tool in new_tools if tool.description not in tool_descriptions
        ]

    def run(
        self,
        query: str,
        llm: BaseLanguageModel,
        verbose: bool = False,
        extra_callbacks: list[BaseCallbackHandler] | None = None,
        extra_metadata: dict | None = None,
    ) -> Tuple[str, pd.DataFrame | None]:
        """Run the maintenance agent."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Make sure to use the provided tool for information.",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Construct the Tools agent
        agent = create_tool_calling_agent(llm, self.tools, prompt)

        extra_callbacks = extra_callbacks or []
        # callbacks = app_conf().callback_handlers + extra_callbacks
        callbacks = extra_callbacks

        metadata = {"agentName": "MaintenanceAgent1"}
        metadata |= extra_metadata or {}

        self._agent_executor = AgentExecutor(
            agent=agent,  # type: ignore
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            # callbacks=callbacks,
            # callbacks=app_conf().callback_handlers,
            metadata=metadata,
        )

        cfg = RunnableConfig()
        cfg["callbacks"] = callbacks
        cfg["metadata"] = metadata
        result = self._agent_executor.invoke({"input": query}, cfg)
        return result["output"]


# python_repl._globals['df'] = df

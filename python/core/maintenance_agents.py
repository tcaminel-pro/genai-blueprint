"""
LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""


import ast
import re
from datetime import datetime
from functools import cached_property
from typing import Any, Literal, Tuple, Union
from textwrap import dedent
from devtools import debug
from loguru import logger
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field


from langchain.agents import (
    AgentType,
    initialize_agent,
    AgentExecutor,
    create_sql_agent,
)
from langchain.tools import Tool, tool, BaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_openai_tools_agent,
)
from langchain import hub
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig
from langchain.utilities.sql_database import truncate_word

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

import streamlit as st

from python.core.dummy_data import DATA_PATH, dummy_database
from python.GenAI_Training import app_conf, llm_agent_test

# Tools setup
PROCEDURES = [
    "procedure_turbine.txt",
    "procedure_generator.txt",
    "procedure_cooling_system.txt",
]


class SqlToolQueryInterceptor(BaseCallbackHandler):
    sql_query: str | None = None
    run_id: str | None = None
    sql_out: str | None = None
    df: pd.DataFrame | None = None

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        if (n := serialized.get("name")) and n == "sql_db_query":
            self.sql_query = input_str
            self.run_id = str(kwargs.get("run_id"))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        if (r := str(kwargs.get("run_id"))) and r == self.run_id and self.sql_query:
            self.str_to_list(output)

    def str_to_list(self, output):
        try:
            self.sql_out = ast.literal_eval(output)
        except:
            logger.warning("unable to parse SQL query output")
        assert self.sql_query
        match = re.search(r"SELECT\s+(.*?)\s+FROM", self.sql_query, re.IGNORECASE)
        if match:
            columns = match.group(1).split(",")
            columns = [col.strip().strip("'").strip('"') for col in columns]
            try:
                self.df = pd.DataFrame(self.sql_out, columns=columns)
            except:
                logger.warning("can't create dataframe from query out")


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
def maintenance_procedure_vectors(text: str) -> Chroma:
    """Index and store a document for similarity matching (through embeddings).

    Embeddings are stored in a vector-store.
    """

    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(DATA_PATH / text))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    return Chroma.from_documents(
        texts, app_conf().embeddings_model, collection_name="maintenance_procedure"
    )


class MaintenanceAgent(BaseModel):
    """Maintenance Agent class"""

    default_llm: BaseLanguageModel
    llm: BaseLanguageModel | None = None
    embeddings_model: Embeddings
    tools: list[BaseTool] = Field(default_factory=list)
    _agent_executor: AgentExecutor | None = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        ignored_types = (cached_property,)

    @property
    def llm_model(self):
        return self.llm if self.llm else self.default_llm

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
        vector_db = Chroma.from_documents(
            few_shot_docs, self.embeddings_model, collection_name="sql_few_shots"
        )
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
            """

        db = SQLDatabase.from_uri(dummy_database())

        from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX

        prefix = (
            SQL_PREFIX
            + f"\nCurrent date is: {datetime.now().isoformat()}. DO NOT use any SQL date functions like NOW(), DATE(), DATETIME()"
        )

        if app_conf().use_functions:
            agent_type = AgentType.OPENAI_FUNCTIONS
        else:
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
        return create_sql_agent(
            llm=self.llm_model,
            toolkit=SQLDatabaseToolkit(db=db, llm=self.llm_model),
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

            qa = RetrievalQA.from_chain_type(
                llm=self.llm_model,
                chain_type="stuff",
                retriever=maintenance_procedure_vectors(PROCEDURES[0]).as_retriever(),
            )
            result = qa.run(full_query)
            return result

        @tool
        def get_info_from_erp(tools_name: list[str], time: str) -> str:
            """Useful to answer questions about tools, such as availability, localization, ...
            Input should be a list of tools for which such information are requested"""

            # Mock function . A real function would call an ERP or PLM API
            result = ""
            for tool in tools_name:
                availability = tool[0].lower() < "r"
                hash = str(tool.__hash__())
                localization = f"R{hash[1:3]}L{hash[3:7]}"
                result += f"{tool}:\n\t- availability: {availability} \n\t- localization:{localization}\n"
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
            return result

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
        llm: BaseLanguageModel | None = None,
        verbose: bool = False,
        extra_callbacks: list[BaseCallbackHandler] | None = None,
        extra_metadata: dict | None = None,
    ) -> Tuple[str, pd.DataFrame | None]:
        """Run the maintenance agent."""

        sql_interceptor = SqlToolQueryInterceptor()
        self.llm = llm if llm else self.default_llm

        if app_conf().use_functions:
            prompt = hub.pull("hwchase17/openai-tools-agent")
            # debug(prompt)
            agent = create_openai_tools_agent(
                prompt=prompt,
                tools=self.tools,
                llm=self.llm_model,
            )
        else:
            prompt = hub.pull("hwchase17/structured-chat-agent")
            debug(prompt)
            agent = create_structured_chat_agent(
                prompt=prompt,
                tools=self.tools,
                llm=self.llm_model,
            )

        extra_callbacks = extra_callbacks or []
        callbacks = app_conf().callback_handlers + extra_callbacks

        metadata = {"agentName": "MaintenanceAgent1"}
        metadata |= extra_metadata or {}

        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            # callbacks=callbacks,
            callbacks=app_conf().callback_handlers,
            metadata=metadata,
        )

        cfg = RunnableConfig()
        cfg["callbacks"] = callbacks
        cfg["metadata"] = metadata
        result = self._agent_executor.invoke({"input": query}, cfg)
        return result["output"], sql_interceptor.df


# python_repl._globals['df'] = df

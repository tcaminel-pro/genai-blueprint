from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing_extensions import Annotated

from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, def_prompt


def create_sql_querying_graph(
    llm: BaseChatModel, db: SQLDatabase, examples: list[dict[str, str]] = []
) -> CompiledGraph:
    class State(TypedDict, total=False):
        question: str
        query: str
        result: str
        answer: str

    class QueryOutput(TypedDict):
        """Generated SQL query."""

        query: Annotated[str, ..., "Syntactically valid SQL query."]

    def write_query(state: State) -> State:
        """Generate SQL query to fetch innformation."""

        system_prompt = dedent_ws(
            """
            Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
            Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. 
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.\n

            Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. 
            Also, pay attention to which column is in which table.\n            
            DO NOT use any SQL date functions like NOW(), DATE(), DATETIME()"

            Only use the following tables:
            {table_info}
            """
        )
        if examples:
            ex = [f"User input: {e['input']}\noutput: {e['query']}" for e in examples]
            system_prompt += (
                "\nBelow are a number of examples of questions and their corresponding SQL queries.\n" + "\n".join(ex)
            )

        user_prompt = "\nCurrent date: {date}.\nQuestion: {input}"
        prompt_template = def_prompt(system_prompt, user_prompt)

        prompt = prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
                "date": datetime.today().strftime("%Y-%B-%d"),
            }
        )
        debug(prompt)
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        assert isinstance(result, dict)
        return {"query": result["query"]}

    def execute_query(state: State) -> State:
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(state: State, config: RunnableConfig) -> State:
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}"
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([write_query, execute_query, generate_answer])
    graph_builder.add_edge(START, "write_query")
    graph = graph_builder.compile()
    return graph


# def sql_querying_chain(query: str, config: RunnableConfig) -> str:
#     llm_id = config["configurable"].get("llm_id")
#     db = config["configurable"].get("db")
#     examples = config["configurable"].get("examples") or []

#     graph = create_sql_querying_graph(llm=get_llm(llm_id), db=db, examples=examples)
#     graph_input = {"messages": [("human", query)]}
#     result = graph.invoke(graph_input)
#     return result["final_response"]


# get_weather_chain = RunnableLambda(get_weather_fn).with_config(llm_config(MODEL) | {"recursion_limit": 6})


if __name__ == "__main__":
    DB_FILE = "use_case_data/other/Chinook.db"
    assert (Path(DB_FILE)).exists()
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    print(db.dialect)
    print(db.get_usable_table_names())
    db.run("SELECT * FROM Artist LIMIT 10;")

    examples = [
        {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
        {
            "input": "Find all albums for the artist 'AC/DC'.",
            "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
        },
        {
            "input": "List all tracks in the 'Rock' genre.",
            "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
        },
        {
            "input": "Find the total duration of all tracks.",
            "query": "SELECT SUM(Milliseconds) FROM Track;",
        },
    ]

    graph = create_sql_querying_graph(get_llm(), db, examples=examples[:5])
    question = "Which country's customers spent the most?"
    for step in graph.stream({"question": "How many employees are there?"}, stream_mode="updates"):
        print(step)

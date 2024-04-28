#
# Adapted from: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/
# See also https://blog.langchain.dev/query-construction/

from functools import cache

from devtools import debug  # ignore
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from python.ai_core.chain_registry import (
    RunnableItem,
    register_runnable,
    to_key_param_callable,
)
from python.ai_core.llm import llm_factory
from python.ai_core.vector_store import vector_store_factory

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]


metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

document_content_description = "Brief summary of a movie"


@cache
def vector_store() -> VectorStore:
    vector_store = vector_store_factory("Chroma_in_memory")
    vector_store.add_documents(docs)
    return vector_store


def get_query_constructor(conf: dict):
    prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
    )
    # debug(prompt.format(query="dummy question"))

    output_parser = StructuredQueryOutputParser.from_components()
    model = conf["llm"]
    llm = llm_factory(model)
    query_constructor = prompt | llm | output_parser
    return query_constructor


def get_retriever(conf: dict):
    query_constructor = get_query_constructor(conf)
    debug(query_constructor)
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,  # TODO: Clarify
        llm_chain=query_constructor,
        vectorstore=vector_store_factory(),
        structured_query_translator=ChromaTranslator(),
    )
    debug(query_constructor, retriever)
    return retriever


register_runnable(
    RunnableItem(
        tag="self_query",
        name="self_query_constructor",
        runnable=to_key_param_callable("query", get_query_constructor),
        examples=[
            "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
        ],
        # key="query",
    )
)


register_runnable(
    RunnableItem(
        tag="self_query",
        name="self_query_retriever",
        runnable=get_retriever,
        examples=[
            "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated",
            "I want to watch a movie rated higher than 8.5",
            "Has Greta Gerwig directed any movies about women",
            "What's a highly rated (above 8.5) science fiction film?",
        ],
    )
)

if __name__ == "__main__":
    r = get_retriever(conf={"llm": None}).invoke(
        "I want to watch a movie rated higher than 8.5"
    )
    debug(r)

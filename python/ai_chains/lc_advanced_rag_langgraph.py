"""
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb
"""

import sys
from operator import itemgetter
from typing import List

from devtools import debug
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from loguru import logger
from typing_extensions import TypedDict

from python.ai_core.chain_registry import (
    RunnableItem,
    register_runnable,
    to_key_param_callable,
)
from python.ai_core.embeddings import embeddings_factory
from python.ai_core.llm import LlmFactory
from python.ai_core.vector_store import document_count, vector_store_factory

# MODEL = "llama3_70_groq"
MODEL = "llama3_8_groq"
llm = LlmFactory(llm_id=MODEL, json_mode=True, cache=True).get()
llm_json = LlmFactory(llm_id=MODEL, json_mode=False, cache=True).get()


def retriever() -> BaseRetriever:
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    vectorstore = vector_store_factory(
        name="Chroma", embeddings=embeddings_factory(), collection_name="rag-chroma"
    )

    if document_count(vectorstore) == 0:
        logger.info("indexing documents...")

        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB

        vectorstore.add_documents(doc_splits)
    retriever = vectorstore.as_retriever()
    return retriever


def retrieval_grader() -> Runnable:
    ### Retrieval Grader
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"
                You are a grader assessing relevance  of a retrieved document to a user question. 
                If the document contains keywords related to the user question, grade it as relevant. 
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            ),
            (
                "user",
                """ 
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n""",
            ),
        ],
    )

    logger.debug("Retrieval Grader'")
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader


def rag_chain() -> Runnable:
    prompt = PromptTemplate(
        template="""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer: """,
        input_variables=["question", "document"],
    )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain


def hallucination_grader() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"
                  You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
                 Give a binary score 'yes' or 'no' score to indicate  whether the answer is grounded in / supported by a set of facts. 
                 Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            ),
            (
                "user",
                """
                    Here are the facts:
                    \n ------- \n
                    {documents} 
                    \n ------- \n
                    Here is the answer: {generation}  
                    """,
            ),
        ],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader


### Answer Grader


def answer_grader() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"
                 You are a grader assessing whether an answer is useful to resolve a question. 
                 Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
                 Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            ),
            (
                "user",
                """
                 Here is the answer:
                \n ------- \n
                {generation} 
                \n ------- \n
                Here is the question: {question} """,
            ),
        ],
    )

    answer_grader = prompt | llm_json | JsonOutputParser()
    return answer_grader


def question_router() -> Runnable:
    prompt = PromptTemplate(
        template="""
        You are an expert at routing a   user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. 
        Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        Return the a JSON with a single key 'datasource' and  no preamble or explanation. 
        Question to route: {question} """,
        input_variables=["question"],
    )

    question_router = prompt | llm_json | JsonOutputParser()
    return question_router


web_search_tool = TavilySearchResults(max_results=3)  # Search tool


# We'll implement these as a control flow in LangGraph.


### State


class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.
    """

    question: str  # the question
    generation: str  # LLM generation
    web_search: str  # whether to add search
    documents: List[Document]  # list of documents


### Nodes


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents from vectorstore
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.debug("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever().invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> GraphState:
    """
    Generate answer using RAG on retrieved documents
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.debug("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain().invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    logger.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader().invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            logger.debug("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            logger.debug("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state: GraphState) -> GraphState:
    """
    Web search based based on the question
    Returns:
        state (dict): Appended web results to documents
    """

    logger.debug("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


### Conditional edge


def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG.
    Returns next node to call
    """

    logger.debug("---ROUTE QUESTION---")
    question = state["question"]
    logger.debug(question)
    source = question_router().invoke({"question": question})
    logger.debug(source)
    logger.debug(source["datasource"])
    if source["datasource"] == "web_search":
        logger.debug("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        logger.debug("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        raise Exception("Bug: unknown source")


def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or add web search
    Returns binary decision for next node to call
    """

    logger.debug("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.debug(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        logger.debug("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the document and answers question.

    Returns: Decision for next node to call
    """

    logger.debug("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader().invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        logger.debug("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        logger.debug("---GRADE GENERATION vs QUESTION---")
        score = answer_grader().invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            logger.debug("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            logger.debug("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        logger.debug("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def create_graph(conf: dict) -> CompiledGraph:
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    ### Graph Build
    # Build graph
    workflow.set_conditional_entry_point(
        route_question,  # type: ignore
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    # Compile
    app = workflow.compile()
    return app


def query_graph(config: dict):
    return create_graph({}) | itemgetter("generation")


register_runnable(
    RunnableItem(
        tag="Advanced RAG",
        name="Advanced-RAG-Langgraph",
        runnable=to_key_param_callable("question", query_graph),
        examples=[
            "What are the types of agent memory",
            "Who are the Bears expected to draft first in the NFL draft?",
        ],
    )
)

ri = RunnableItem(
    tag="Advanced RAG",
    name="Advanced-RAG-Langgraph",
    runnable=to_key_param_callable("question", query_graph),
    examples=[
        "What are the types of agent memory",
        "Who are the Bears expected to draft first in the NFL draft?",
    ],
)


# Test


def test_graph_stream():
    app = create_graph({})
    inputs = {"question": "What are the types of agent memory?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            logger.info(f"Finished running: {key}:")
    print(value["generation"])

    inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            logger.info(f"Finished running: {key}:")
    print(value["generation"])

    # https://smith.langchain.com/public/c785f9c0-f519-4a38-ad5a-febb59a2139c/r
    app.get_graph().draw_png("test_graph")


def test_graph():
    chain = query_graph({}) | itemgetter("generation")
    input = {"question": "What are the types of agent memory?"}
    r = chain.invoke(input)
    debug(r)


def test_nodes():
    question = "agent memory"
    docs = retriever().invoke(question)
    doc_txt = docs[1].page_content
    debug(retrieval_grader().invoke({"question": question, "document": doc_txt}))

    # Run
    docs = retriever().invoke(question)
    generation = rag_chain().invoke({"context": docs, "question": question})
    debug(generation)
    ### Hallucination Grader

    hallucination = hallucination_grader().invoke(
        {"documents": docs, "generation": generation}
    )
    debug(hallucination)

    answer = answer_grader().invoke({"question": question, "generation": generation})
    debug(answer)

    question = "llm agent memory"
    docs = retriever().invoke(question)
    doc_txt = docs[1].page_content
    logger.debug(question_router().invoke({"question": question}))


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<blue>{level}</blue> | <green>{message}</green>",
        colorize=True,
    )
    # test_nodes()
    test_graph()

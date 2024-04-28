"""
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb
"""

from typing import List

from devtools import debug
from langchain import hub
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict

from python.ai_core.embeddings import embeddings_factory
from python.ai_core.llm import LlmFactory
from python.ai_core.vector_store import vector_store_factory

MODEL ="llama3_70_groq"
llm= LlmFactory(llm_id=MODEL, json_mode=True).get()
llm_json= LlmFactory(llm_id=MODEL, json_mode=False).get()

def retriever() : 
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = vector_store_factory(name = "Chroma", embeddings=embeddings_factory(), collection_name="rag-chroma")                                     
    vectorstore.add_documents(doc_splits)
    retriever = vectorstore.as_retriever()
    return retriever

def retrieval_grader (): 
    ### Retrieval Grader 
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """"
                You are a grader assessing relevance  of a retrieved document to a user question. 
                If the document contains keywords related to the user question, grade it as relevant. 
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
                ("user", """ 
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n"""),
            ],
        )

    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader


def rag_chain() :
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

def hallucination_grader() : 
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """"
                  You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
                 Give a binary score 'yes' or 'no' score to indicate  whether the answer is grounded in / supported by a set of facts. 
                 Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
                 ("user", """
                    Here are the facts:
                    \n ------- \n
                    {documents} 
                    \n ------- \n
                    Here is the answer: {generation}  
                    """
                  ) 
            ]
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

### Answer Grader 

def answer_grader():
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """"
                 You are a grader assessing whether an answer is useful to resolve a question. 
                 Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
                 Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
                 ("user", """
                 Here is the answer:
                \n ------- \n
                {generation} 
                \n ------- \n
                Here is the question: {question} """)],
    )

    answer_grader = prompt | llm_json| JsonOutputParser()
    return answer_grader

def question_router():

    prompt = PromptTemplate(
        template="""
        You are an expert at routing a   user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. 
        Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        Return the a JSON with a single key 'datasource' and  no premable or explaination. 
        Question to route: {question} """,
        input_variables=["question"],
    )

    question_router = prompt | llm_json | JsonOutputParser()
    return question_router


### Search

web_search_tool = TavilySearchResults(k=3)


#We'll implement these as a control flow in LangGraph.


### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]



### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
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

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})  
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae
### Graph Build
# Build graph
workflow.set_conditional_entry_point(
    route_question,
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

# Test
from pprint import pprint

inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
Trace: 


# Compile
app = workflow.compile()

# Test
from pprint import pprint

inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
Trace: 

https://smith.langchain.com/public/c785f9c0-f519-4a38-ad5a-febb59a2139c/r
app.get_graph().print_ascii()

def test() :
    
    question = "agent memory"
    docs = retriever().invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader().invoke({"question": question, "document": doc_txt}))

    # Run
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)
    ### Hallucination Grader 

    hallucination_grader().invoke({"documents": docs, "generation": generation})

    answer_grader().invoke({"question": question,"generation": generation})

    
    question = "llm agent memory"
    docs = retriever().get_relevant_documents(question)
    doc_txt = docs[1].page_content
    print(question_router().invoke({"question": question}))




#
# Taken from


from functools import cache
import sys
from pathlib import Path
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from loguru import logger
from devtools import debug


from python.ai_core.chain_registry import RunnableItem, register_runnable
from python.config import get_config
from python.ai_core.embeddings import embeddings_factory
from python.ai_core.llm import llm_getter
from python.ai_core.vector_store import vector_store_factory


base_dir = Path(get_config("documents", "base"))
assert base_dir.exists()


@cache
def retriever():
    dir = base_dir / "maintenance"
    vector_store = vector_store_factory("Chroma", embeddings_factory(), "maintenance_1")

    found = vector_store.similarity_search("maintenance", k=3)
    if len(found) == 0:
        logger.info(f"retrieving text document in {dir} to VectorStore")
        loader = DirectoryLoader(str(dir), glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=splits)

    return vector_store.as_retriever()


prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_getter()
    | StrOutputParser()
)


register_runnable(
    RunnableItem(
        tag="RAG",
        name="Simple RAG chain on Maintenance procedures",
        runnable=rag_chain,
        examples=[
            "What are the tools required for the maintenance task 'Repair of Faulty Switchgear' "
        ],
    )
)
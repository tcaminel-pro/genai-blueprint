#
# Taken from


import sys
from pathlib import Path
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from devtools import debug
from loguru import logger


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.config import get_config


from python.ai_core.embeddings import get_embeddings
from python.ai_core.llm import get_llm, set_cache
from python.ai_core.vector_store import get_vector_store


set_cache()

base_dir = Path(get_config("documents", "base") )

def retriever():
    dir = base_dir / "maintenance"
    logger.info(f"retrieving text document in {dir} to VectorStore")
    loader = DirectoryLoader(dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = get_vector_store("Chroma", get_embeddings(), "test")
    vectorstore.add_documents(documents=splits)

    return vectorstore.as_retriever()


prompt = hub.pull("rlm/rag-prompt")
llm = get_llm()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# debug(rag_chain.invoke("What are the Required Tools for the task  'Visual Inspection of Generators'  ?"))

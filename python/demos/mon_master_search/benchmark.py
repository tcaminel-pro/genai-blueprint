from functools import cache
from pathlib import Path

import pandas as pd
import spacy
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import ConfigurableField, Runnable

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.loaders import load_docs_from_jsonl
from python.ai_core.vector_store import VectorStoreFactory

DEFAULT_RESULT_COUNT = 15
RATIO_SPARSE = 60
EMBEDDINGS_MODEL_ID = "solon-large"
REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
FILES = REPO / "synthesis.json"


@cache
def get_sparse_retriever(embeddings_model_id: str) -> Runnable:
    embeddings_factory = EmbeddingsFactory(embeddings_id=embeddings_model_id)
    retriever = VectorStoreFactory(
        id="Chroma",
        embeddings_factory=embeddings_factory,
        collection_name="offres_formation",
    ).get_configurable_retriever(default_k=DEFAULT_RESULT_COUNT)
    return retriever


@cache
def spacy_model(model="fr_core_news_sm") -> tuple[spacy.language.Language, set[str]]:
    nlp = spacy.load(model)
    stop_words = nlp.Defaults.stop_words
    stop_words.update(
        {",", ";", "(", ")", ":", "[", "]", "master", "mastère", "formation", "\n"}
    )
    return nlp, stop_words


def preprocess_text(text) -> list[str]:
    nlp, stop_words = spacy_model()
    lemmas = [token.lemma_.lower() for token in nlp(text)]
    filtered = [token for token in lemmas if token not in stop_words]
    return filtered


@cache
def get_bm25_retriever() -> Runnable:
    docs_for_bm25 = load_docs_from_jsonl(FILES)
    retriever = BM25Retriever.from_documents(
        documents=docs_for_bm25,
        preprocess_func=preprocess_text,
        k=DEFAULT_RESULT_COUNT,
    ).configurable_fields(k=ConfigurableField(id="k"))
    return retriever


@cache
def get_ensemble_retriever(
    embeddings_model_id: str, ratio_sparse: float
) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(), get_sparse_retriever(embeddings_model_id)],
        weights=[1.0 - ratio_sparse, ratio_sparse],
    )


def search(query: str) -> pd.DataFrame:
    known_set: set[str] = set()
    df = pd.DataFrame()
    retriever = get_ensemble_retriever(EMBEDDINGS_MODEL_ID, ratio_sparse=RATIO_SPARSE)

    #  quick and dirty hack to have enough results
    count = DEFAULT_RESULT_COUNT * 2

    config = {"configurable": {"k": count, "search_kwargs": {"k": count}}}
    user_input = "query : " + query  # supposed to work well for Solon Embeddings

    result = retriever.invoke(user_input, config=config)  # type: ignore

    for doc in result:
        # obj = json.loads(doc.page_content)
        intitule = doc.page_content.removeprefix("intitulé: ")
        eta = doc.metadata.get("eta_name")
        inm = doc.metadata.get("source")
        for_intitule = doc.metadata["for_intitule"]

        if for_intitule in known_set:
            pass
        else:
            row = {
                "Intitulé formation:": for_intitule,
                "Intitulé parcours ou DMM": intitule,
                "ETA": eta,
                "INM(P)": inm,
                "Content": doc.page_content,
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        known_set.add(for_intitule)

    return df
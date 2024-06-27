from enum import Enum
from functools import cache
from pathlib import Path

import pandas as pd
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import Runnable
from loguru import logger

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.loaders import load_docs_from_jsonl
from python.ai_core.vector_store import VectorStoreFactory
from python.ai_retrievers.bm25s_retriever import (
    get_spacy_preprocess_fn,
)
from python.demos.mon_master_search.model_subset import EXAMPLE_QUERIES

DEFAULT_RESULT_COUNT = 20
RATIO_SPARSE = 50
EMBEDDINGS_MODEL_ID = "solon-large"
REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
FILES = REPO / "synthesis_v2.json"


class SearchMode(Enum):
    VECTOR = "Vector"
    KEYWORD = "Keyword"
    HYBRID = "Hybrid"


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
def get_bm25_retriever():
    stop_words = [
        "master",
        "mastère",
        "formation",
        "diplome",
    ]
    fn = get_spacy_preprocess_fn(model="fr_core_news_sm", more_stop_words=stop_words)  # noqa: F821
    logger.info("create BM25 index")
    docs_for_bm25 = list(load_docs_from_jsonl(FILES))
    docs = docs_for_bm25

    retriever = BM25Retriever.from_documents(
        documents=docs, preprocess_func=fn, k=DEFAULT_RESULT_COUNT
    )
    # path = Path(get_config_str("vector_store", "path")) / "bm25"
    # retriever = BM25FastRetriever.from_cache(
    #     preprocess_func=fn,
    #     k=DEFAULT_RESULT_COUNT,
    # )
    return retriever


def get_ensemble_retriever(
    embeddings_model_id: str, ratio_sparse: float
) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(), get_sparse_retriever(embeddings_model_id)],
        weights=[1.0 - ratio_sparse, ratio_sparse],
    )


def search(
    query: str, mode: SearchMode = SearchMode.VECTOR, ratio: int = RATIO_SPARSE
) -> pd.DataFrame:
    known_set: set[str] = set()
    df = pd.DataFrame()
    if mode == SearchMode.VECTOR:
        retriever = get_sparse_retriever(EMBEDDINGS_MODEL_ID)
    elif mode == SearchMode.KEYWORD:
        retriever = get_bm25_retriever()
    else:
        retriever = get_ensemble_retriever(EMBEDDINGS_MODEL_ID, ratio_sparse=ratio)

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

        key = doc.metadata["for_intitule"] + doc.metadata["eta_uai"]
        parcours = intitule.partition('" : ')[2]
        parcours = parcours.replace("parcours: ", "=> ")
        row = {
            "Intitulé formation:": for_intitule,
            "Parcours": parcours,
            "ETA": eta,
            "INM(P)": inm,
            "Content": doc.page_content,
        }
        if key not in known_set:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        known_set.add(key)

    return df


# cSpell: disable
def process_questions(
    queries: list[str], mode: SearchMode = SearchMode.VECTOR, ratio: int = RATIO_SPARSE
) -> list[dict]:
    result = []
    for q in queries:
        df = search(q, mode, ratio)
        d = {"question": q}
        i = 1
        for answer in df.itertuples():
            line = f"{answer[3]}: {answer[1]} ({answer[4]})"
            d |= {i: line}
            i += 1
        result.append(d)
    return result


def format_sheet(worksheet):
    for column in worksheet.columns:
        column_letter = column[0].column_letter  # Get the column letter
        if column_letter > "A":
            worksheet.column_dimensions[column_letter].width = 100


if __name__ == "__main__":
    # cSpell: disable

    OUT_FILE = REPO / "master_search_v0_5.xlsx"

    logger.info(f"write Exel file : {OUT_FILE}")
    with pd.ExcelWriter(OUT_FILE) as writer:
        logger.info("Vector Search (Solon-large)...")
        d_vector = process_questions(EXAMPLE_QUERIES, SearchMode.VECTOR)
        pd.DataFrame(d_vector).to_excel(
            writer, sheet_name="Vector_search", freeze_panes=(0, 2)
        )
        format_sheet(writer.sheets["Vector_search"])

        logger.info("Hybrid Search 50/50...")
        d_hybrid = process_questions(_questions, SearchMode.HYBRID, 50)
        sheet = "Hybrid_search_50_50"
        pd.DataFrame(d_hybrid).to_excel(writer, sheet_name=sheet, freeze_panes=(0, 2))
        format_sheet(writer.sheets[sheet])

        # logger.info("Hybrid Search 70/30...")
        # d_hybrid = process_questions(_questions, SearchMode.HYBRID, 70)
        # sheet = "Hybrid_search_70_30"
        # pd.DataFrame(d_hybrid).to_excel(writer, sheet_name=sheet, freeze_panes=(0, 2))
        # format_sheet(writer.sheets[sheet])

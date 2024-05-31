"""
Demo of Semantic Search

Copyright (C) 2024 Eviden. All rights reserved
"""

# cSpell: disable

from pathlib import Path

import pandas as pd
import spacy
import streamlit as st
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import ConfigurableField, Runnable

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.loaders import load_docs_from_jsonl
from python.ai_core.vector_store import VectorStoreFactory
from python.demos.mon_master_search.model_subset import (
    LICENCES_CONSEILLEES,
    MODELITE_ENSEIGNEMENT,
)

LLM = "gemini_pro_google"

REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
FILES = REPO / "synthesis.json"

DEFAULT_RESULT_COUNT = 25


################################
#  UI
################################

st.set_page_config(layout="wide")

title_col1, title_col2 = st.columns([2, 1])

logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")

st.logo(logo_eviden)

title_col1.title("Recherche Sémantique de Masters")
title_col2.image(logo_eviden, width=250)
title_col1.markdown(
    """
    Démo
    """,
    unsafe_allow_html=True,
)

filter1, filter2, filter3 = st.columns(3)
filter1.multiselect("licence", LICENCES_CONSEILLEES)
filter2.multiselect("modalité", MODELITE_ENSEIGNEMENT)
filter3.multiselect("Villes", [])

with st.sidebar:
    search_method = st.radio("Search Method", ["Vector", "Keyword", "Hybrid"])
    if search_method == "Hybrid":
        ratio_spinner = st.slider(
            "Keyword  / Vector ratio", min_value=0.0, max_value=1.0, value=0.7, step=0.1
        )

    embeddings_model = st.radio(
        "Embedding Model",
        options=["multilingual_MiniLM_local", "camembert_large_local"],
        captions=["Small multilingual model", "Large model for French"],
    )
    result_count = int(
        st.number_input(
            "Nombre de parcours recherchés",
            min_value=1,
            max_value=60,
            value=DEFAULT_RESULT_COUNT,
        )
    )
    show_dmm_only = st.toggle("Regrouper par mention", False)


with st.form(key="form"):
    user_input = st.text_area(label="Recherche:", value="", height=150)
    submit_clicked = st.form_submit_button("Rechercher")


@st.cache_resource(show_spinner="Load vector store...")
def get_sparse_retriever(embeddings_model_id: str) -> Runnable:
    embeddings_factory = EmbeddingsFactory(embeddings_id=embeddings_model_id)
    vector_store = VectorStoreFactory(
        id="Chroma",
        embeddings_factory=embeddings_factory,
        collection_name="offres_formation",
    ).vector_store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": DEFAULT_RESULT_COUNT}
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
        )
    )
    return retriever


@st.cache_resource(show_spinner="load NLP model...")
def spacy_model(model="fr_core_news_sm"):
    nlp = spacy.load(model)
    stop_words = nlp.Defaults.stop_words
    stop_words.update({",", ";", "(", ")", ":", "[", "]"})
    return nlp, stop_words


def preprocess_text(text) -> list[str]:
    nlp, stop_words = spacy_model()
    lemmas = [token.lemma_.lower() for token in nlp(text)]
    filtered = [token for token in lemmas if token not in stop_words]
    debug(filtered)
    return filtered


@st.cache_resource(show_spinner="index documents for keyword search...")
def get_bm25_retriever() -> Runnable:
    docs_for_bm25 = []
    for doc in load_docs_from_jsonl(FILES):
        docs_for_bm25.append(
            Document(
                page_content=str(doc.metadata["for_intitule"]), metadata=doc.metadata
            )
        )

    retriever = BM25Retriever.from_documents(
        documents=docs_for_bm25,
        preprocess_func=preprocess_text,
        k=DEFAULT_RESULT_COUNT,
    ).configurable_fields(k=ConfigurableField(id="k"))
    return retriever


@st.cache_resource(show_spinner="load hybrid model...")
def get_ensemble_retriever(
    embeddings_model_id: str, ratio_sparse: float
) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(), get_sparse_retriever(embeddings_model_id)],
        weights=[1.0 - ratio_sparse, ratio_sparse],
    )


def regroup_results(docs: list[Document]):
    d = {}
    for doc in docs:
        for_intitule = doc.metadata["for_intitule"]
        if not d.get("for_intitule"):
            d |= {for_intitule: doc}


df = pd.DataFrame()
knwon_set: set[str] = set()
if submit_clicked:
    assert embeddings_model is not None
    if search_method == "Vector":
        retriever = get_sparse_retriever(embeddings_model)
    elif search_method == "Keyword":
        retriever = get_bm25_retriever()
    elif search_method == "Hybrid":
        retriever = get_ensemble_retriever(embeddings_model, ratio_sparse=ratio_spinner)

    count = result_count * 2 if show_dmm_only else result_count  #  quick and dirty hack

    config = {"configurable": {"k": count, "search_kwargs": {"k": count}}}
    result = retriever.invoke(user_input, config=config)  # type: ignore
    for doc in result:
        # obj = json.loads(doc.page_content)
        intitule = doc.page_content.removeprefix("intitulé: ")
        eta = doc.metadata.get("eta_name")
        inm = doc.metadata.get("source")
        for_intitule = doc.metadata["for_intitule"]

        if show_dmm_only and for_intitule in knwon_set:
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
        knwon_set.add(for_intitule)

    st.dataframe(df)

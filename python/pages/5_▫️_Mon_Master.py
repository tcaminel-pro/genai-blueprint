"""
Demo of Semantic Search

Copyright (C) 2024 Eviden. All rights reserved
"""

from pathlib import Path

import pandas as pd
import spacy
import streamlit as st
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.loaders import load_docs_from_jsonl
from python.ai_core.vector_store import VectorStoreFactory
from python.demos.mon_master_search.model_subset import (
    LICENCES_CONSEILLEES,
    MODELITE_ENSEIGNEMENT,
)

# cSpell: disable

LLM = "gemini_pro_google"


################################
#  UI
################################

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

# EMBEDDINGS_MODEL = "multilingual_MiniLM_local"
EMBEDDINGS_MODEL = "camembert_large_local"


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


with st.form(key="form"):
    user_input = st.text_area(label="Recherche:", value="", height=150)
    submit_clicked = st.form_submit_button("Submit Question")


@st.cache_resource(show_spinner="Load vector store...")
def get_sparse_retriever(embeddings_model_id: str) -> VectorStoreRetriever:
    embeddings_factory = EmbeddingsFactory(embeddings_id=embeddings_model_id)
    vector_store = VectorStoreFactory(
        id="Chroma",
        embeddings_factory=embeddings_factory,
        collection_name="offres_formation",
    ).vector_store
    return vector_store.as_retriever(search_kwargs={"k": 20})


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
    return filtered


@st.cache_resource(show_spinner="index documents for keyword search...")
def get_bm25_retriever() -> BM25Retriever:
    REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
    FILES = REPO / "synthesis.json"

    docs_for_bm25 = []
    for doc in load_docs_from_jsonl(FILES):
        docs_for_bm25.append(
            Document(page_content=str(doc.metadata["title"]), metadata=doc.metadata)
        )

    return BM25Retriever.from_documents(
        documents=docs_for_bm25,
        preprocess_func=preprocess_text,
        k=20,
    )


@st.cache_resource(show_spinner="load hybrid model...")
def get_ensemble_retriever(
    embeddings_model_id: str, ratio_sparse: float
) -> EnsembleRetriever:
    debug([1.0 - ratio_sparse, ratio_sparse])
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(), get_sparse_retriever(embeddings_model_id)],
        weights=[1.0 - ratio_sparse, ratio_sparse],
    )


df = pd.DataFrame()
if submit_clicked:
    assert embeddings_model is not None
    if search_method == "Vector":
        retriever = get_sparse_retriever(embeddings_model)
    elif search_method == "Keyword":
        retriever = get_bm25_retriever()
    elif search_method == "Hybrid":
        retriever = get_ensemble_retriever(embeddings_model, ratio_sparse=ratio_spinner)

    result = retriever.invoke(user_input)

    for doc in result:
        # obj = json.loads(doc.page_content)
        intitule = doc.page_content.removeprefix("intitulé: ")
        eta = doc.metadata.get("eta_name")
        inm = doc.metadata.get("source")

        row = {
            "intitulé": intitule,
            "eta": eta,
            "inm": inm,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    st.dataframe(df)

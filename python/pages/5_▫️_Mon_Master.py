"""Demo of Semantic Search


"""

# cSpell: disable

import timeit
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import Runnable

import python.demos.mon_master_search.search as master_search
from python.demos.mon_master_search.loader import add_accronym
from python.demos.mon_master_search.model_subset import EXAMPLE_QUERIES

LLM = "gemini_pro_google"

REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
FILES = REPO / "synthesis_v2.json"

DEFAULT_RESULT_COUNT = 100


################################
#  UI
################################

st.set_page_config(layout="wide")

title_col1, title_col2 = st.columns([2, 1])

logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")

st.logo(logo_eviden)

title_col1.title("Recherche Sémantique de Masters")
title_col2.image(logo_eviden, width=250)


# filter1, filter2, filter3 = st.columns(3)
# filter1.multiselect("licence", LICENCES_CONSEILLEES)
# filter2.multiselect("modalité", MODELITE_ENSEIGNEMENT)
# filter3.multiselect("Villes", [])

with st.sidebar:
    search_method = st.radio("Search Method:", ["Vector", "Keyword", "Hybrid"], index=2)
    if search_method == "Hybrid":
        ratio_spinner = st.slider("Keyword  / Vector ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    embeddings_model = st.radio(
        "Embedding Model:",
        options=[
            "multilingual_MiniLM_local",
            "camembert_large_local",
            "solon-large",
        ],
        captions=[
            "Small multilingual model",
            "Large model for French",
            "SOTA model for French",
        ],
        index=2,
    )
    result_count = int(
        st.number_input(
            "Nombre de parcours recherchés",
            min_value=1,
            max_value=100,
            value=DEFAULT_RESULT_COUNT,
        )
    )

example = st.selectbox("Examples:", EXAMPLE_QUERIES, index=None)
with st.form(key="form"):
    user_input = st.text_area(label="Recherche:", value=example or "", height=70)
    submit_clicked = st.form_submit_button("Rechercher")


@st.cache_resource(show_spinner="Load vector store...")
def _get_sparse_retriever(embeddings_model_id: str) -> Runnable:
    return master_search.get_sparse_retriever(embeddings_model_id)


@st.cache_resource(show_spinner="index documents for keyword search...")
def _get_bm25_retriever() -> Runnable:
    return master_search.get_bm25_retriever()


@st.cache_resource(show_spinner="load hybrid model...")
def get_ensemble_retriever(embeddings_model_id: str, ratio_sparse: float) -> EnsembleRetriever:
    return master_search.get_ensemble_retriever(embeddings_model_id, ratio_sparse)


df = pd.DataFrame()
known_set: set[str] = set()
if submit_clicked:
    assert embeddings_model is not None
    if search_method == "Vector":
        retriever = _get_sparse_retriever(embeddings_model)
    elif search_method == "Keyword":
        retriever = _get_bm25_retriever()
    elif search_method == "Hybrid":
        retriever = get_ensemble_retriever(embeddings_model, ratio_sparse=ratio_spinner)

    #  quick and dirty hack to have enough results
    count = result_count * 2

    config = {"configurable": {"k": count, "search_kwargs": {"k": count}}}
    user_input = "query : " + add_accronym(user_input)  # supposed to work well for Solon Embeddings

    start_time = timeit.default_timer()
    result = retriever.invoke(user_input, config=config)  # type: ignore
    #    debug(user_input, result)
    delta_t = timeit.default_timer() - start_time
    for doc in result:
        # obj = json.loads(doc.page_content)
        intitule = doc.page_content.removeprefix("intitulé: ")
        eta = doc.metadata.get("eta_name")
        inm = doc.metadata.get("source")
        for_intitule = doc.metadata["for_intitule"]
        key = doc.metadata["for_intitule"] + doc.metadata["eta_uai"]
        parcours = intitule.partition('" : ')[2]
        parcours = parcours.replace("parcours: ", "=> ").replace("libelés:", " -- ")

        row = {
            "Intitulé formation:": for_intitule,
            "Formation => Parcours -- libelés": parcours.lower(),
            "ETA": eta,
            "INM(P)": inm,
            "Content": doc.page_content,
        }
        if key not in known_set:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        known_set.add(key)

    st.dataframe(df)
    st.write(f"search duration : {delta_t:9.5f} s")

"""
A replacement of the BM25Retriever with , much faster
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from loguru import logger


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


def get_spacy_preprocess_fn(model: str, more_stop_words: list[str] = []):
    import spacy

    logger.info(f"load spacy model {model}")

    nlp = spacy.load(model)
    stop_words = nlp.Defaults.stop_words
    stop_words.update(more_stop_words or [])

    def preprocess_text(text) -> list[str]:
        lemmas = [token.lemma_.lower() for token in nlp(text)]
        filtered = [token for token in lemmas if token not in stop_words]
        return filtered

    return preprocess_text


class BM25FastRetriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        cache_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> BM25FastRetriever:
        """
        Create a BM25S_Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25S_Retriever instance.
        """
        try:
            import bm25s
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install bm25s"
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = bm25s.BM25(**bm25_params)
        vectorizer.index(texts_processed, show_progress=True)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        if cache_path is not None:
            logger.info("Save BM25 cache")
            vectorizer.save(cache_path, corpus=None, allow_pickle=True)

        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> BM25FastRetriever:
        """
        Create a BM25S_Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25S_Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            cache_path=cache_dir,
            **kwargs,
        )

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25FastRetriever:
        try:
            import bm25s
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install bm25s"
            )

        logger.info("Load BM25 cache")
        vectorizer = bm25s.BM25.load(cache_dir, mmap=False, allow_pickle=True)
        return cls(
            vectorizer=vectorizer, preprocess_func=preprocess_func, docs=[], **kwargs
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        # return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        results = self.vectorizer.retrieve(
            processed_query, corpus=self.docs, k=self.k, return_as="documents"
        )
        return_docs = [results[0, i] for i in range(results.shape[1])]
        logger.debug(f"search : {query=} {return_docs=}")
        return return_docs

"""A replacement of the BM25Retriever, much faster.

It uses the BM25s package, and SpaCy for the preprocessing."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import ConfigDict, Field


def default_preprocessing_func(text: str) -> list[str]:
    return text.split()


def get_spacy_preprocess_fn(model: str, more_stop_words: list[str] | None = None) -> Callable[[str], list[str]]:
    """Return a function that preprocess a string for search :  lemmanisation, lower_case, and strop word removal.

    Args:
    - model: spacy model for lemmanisation
    - more_stop_words : additional stop words
    """
    import spacy

    if more_stop_words is None:
        more_stop_words = []
    logger.info(f"load spacy model {model}")

    try:
        nlp = spacy.load(model)
    except OSError as ex:
        raise ModuleNotFoundError(
            f"Cannot load Spacy model.  Try install it with : 'python -m spacy download {model}'"
        ) from ex

    stop_words = nlp.Defaults.stop_words
    stop_words.update(more_stop_words or [])

    def preprocess_text(text: str) -> list[str]:
        lemmas = [token.lemma_.lower() for token in nlp(text)]
        filtered = [token for token in lemmas if token not in stop_words]
        return filtered

    return preprocess_text


class BM25FastRetriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: list[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], list[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[dict[str, Any]] = None,
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func,
        cache_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> "BM25FastRetriever":
        """Create a BM25S_Retriever from a list of texts.

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
            import bm25s  # type: ignore
        except ImportError as ex:
            raise ImportError("Could not import bm25s, please install with `pip install bm25s") from ex

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = bm25s.BM25(**bm25_params)
        vectorizer.index(texts_processed, show_progress=True)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas, strict=False)]

        if cache_path is not None:
            logger.info("Save BM25 cache")
            vectorizer.save(cache_path, corpus=None, allow_pickle=True)

        return cls(vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[dict[str, Any]] = None,
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> "BM25FastRetriever":
        """Create a BM25S_Retriever from a list of Documents.

        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25S_Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents), strict=False)
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            cache_path=cache_dir,
            **kwargs,
        )

    @classmethod
    def from_index_file(
        cls,
        index_file: Path,
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25FastRetriever":
        try:
            import bm25s  # type: ignore
        except ImportError as ex:
            raise ImportError("Could not import bm25s, please install with `uv add bm25s`") from ex

        logger.info("Load BM25 index")
        vectorizer = bm25s.BM25.load(index_file, mmap=False, allow_pickle=True)

        # Load documents from cache if available
        docs = []
        if hasattr(vectorizer, "corpus") and vectorizer.corpus is not None:
            docs = [Document(page_content=text, metadata={}) for text in vectorizer.corpus]

        return cls(vectorizer=vectorizer, preprocess_func=preprocess_func, docs=docs, **kwargs)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        processed_query = self.preprocess_func(query)

        if not self.docs:
            # If docs is empty, return empty list
            logger.warning("No documents available for retrieval")
            return []

        results = self.vectorizer.retrieve([processed_query], corpus=self.docs, k=self.k, return_as="documents")
        return_docs = [results[0, i] for i in range(results.shape[1])]
        logger.debug(f"search : {query=} {return_docs=}")
        return return_docs


if __name__ == "__main__":
    """Quick test for BM25FastRetriever using SpaCyModelManager and spacy preprocessing."""
    from src.utils.spacy_model_mngr import SpaCyModelManager

    # Use SpaCyModelManager to handle spacy model
    model_name = "en_core_web_sm"  # Default model name
    SpaCyModelManager().setup_spacy_model(model_name)

    # Get spacy preprocessing function
    preprocess_func = get_spacy_preprocess_fn(model_name)

    sample_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over lazy dogs in summer.",
        "The lazy dog sleeps in the sun.",
        "Machine learning algorithms process information efficiently.",
        "Artificial intelligence models require substantial training data.",
    ]
    metadatas = [{"id": i, "source": "test"} for i in range(len(sample_docs))]

    # Create retriever with spacy preprocessing
    retriever = BM25FastRetriever.from_texts(
        texts=sample_docs,
        metadatas=metadatas,
        preprocess_func=preprocess_func,
        k=3,
    )

    # Test retrieval
    queries = ["fox", "machine learning", "artificial intelligence"]

    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content} (metadata: {doc.metadata})")

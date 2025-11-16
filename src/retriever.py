from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .config import (
    MAX_CONTEXT_CHUNKS,
    RETRIEVER_K,
    RERANKER_DEVICE,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_K,
    USE_RERANKER,
)
from .vector_store import get_vector_store


@lru_cache(maxsize=1)
def _get_vectorstore():
    """Cache the vector store to avoid re-creating it (saves time & memory)."""
    return get_vector_store()


def get_base_retriever():
    """Base retriever from Chroma using similarity search."""
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


@lru_cache(maxsize=1)
def _get_reranker_model() -> CrossEncoder:
    """Create and cache the bge-reranker (cross-encoder) model.

    We wrap it in lru_cache to ensure the model is loaded only once per process,
    avoiding multiple copies in RAM.
    """
    model = CrossEncoder(RERANKER_MODEL_NAME, device=RERANKER_DEVICE)
    return model


def _rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    """Rerank the given Documents by relevance to the query using bge-reranker."""
    if not docs:
        return []

    model = _get_reranker_model()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    limit = top_k or RERANKER_TOP_K or MAX_CONTEXT_CHUNKS
    limit = min(limit, len(scored_docs))
    reranked = [doc for doc, _ in scored_docs[:limit]]
    return reranked


def retrieve_documents(
    query: str,
    use_reranker: bool | None = None,
) -> List[Document]:
    """High-level helper to retrieve relevant documents for a query.

    - Step 1: use the base retriever (Chroma similarity search) to get k docs.
    - Step 2 (optional): rerank them with bge-reranker and keep the top_k docs.
    """
    use_reranker = USE_RERANKER if use_reranker is None else use_reranker

    retriever = get_base_retriever()
    docs: List[Document] = retriever.invoke(query)

    if use_reranker:
        docs = _rerank_documents(query, docs)

    return docs



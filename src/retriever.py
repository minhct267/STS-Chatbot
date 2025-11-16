from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .config import (MAX_CONTEXT_CHUNKS, RERANKER_DEVICE, RERANKER_MODEL_NAME,
                     RERANKER_TOP_K, RETRIEVER_K, USE_RERANKER)
from .vector_store import get_vector_store


@lru_cache(maxsize=1)
def _get_vectorstore():
    return get_vector_store()


def get_base_retriever():
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


@lru_cache(maxsize=1)
def _get_reranker_model() -> CrossEncoder:
    model = CrossEncoder(RERANKER_MODEL_NAME, device="cuda")
    return model


def _rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int | None = None,
) -> List[Document]:
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
    use_reranker = USE_RERANKER if use_reranker is None else use_reranker

    retriever = get_base_retriever()
    docs: List[Document] = retriever.invoke(query)

    if use_reranker:
        docs = _rerank_documents(query, docs)

    return docs

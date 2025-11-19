from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from .config import (
    DEFAULT_TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
    MIN_RELEVANCE_SCORE_FOR_CONTEXT,
    OLLAMA_MODEL_NAME,
)
from .prompt import get_general_chat_prompt_template, get_rag_prompt_template
from .retriever import retrieve_documents, retrieve_documents_with_scores


def _format_docs(docs: List[Document]) -> str:
    lines: List[str] = []
    for idx, doc in enumerate(docs[:MAX_CONTEXT_CHUNKS], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header = f"[{idx}] Source: {source}"
        if page is not None:
            header += f" (page {page})"
        lines.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(lines)


def get_rag_chain(
    use_reranker: bool = True,
    model_name: str | None = None,
    temperature: float | None = None,
) -> Any:
    prompt = get_rag_prompt_template()
    llm = ChatGroq(
        model=model_name or OLLAMA_MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
    )

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(
    question: str,
    use_reranker: bool = True,
    model_name: str | None = None,
    temperature: float | None = None,
    chat_history: str | None = None,
) -> Dict[str, Any]:
    """Answer a user question, with intelligent fallback when retrieval is weak.

    Behaviour:
    - If the vector store returns sufficiently relevant documents, we use a
      document-grounded RAG prompt and include those documents as context.
    - If no documents are found, or all similarity scores are too low, we fall
      back to a general chat prompt that answers using broad world knowledge
      (suitable for small-talk, weather questions, etc.).
    """
    chat_history_text = chat_history or ""

    # Retrieve documents together with similarity scores so we can decide
    # whether there is enough signal to trust the context.
    docs, scores = retrieve_documents_with_scores(
        question,
        use_reranker=use_reranker,
    )
    max_score = max(scores) if scores else None

    has_docs = bool(docs)
    # If the vector store does not expose scores, we treat the presence of any
    # documents as a positive retrieval signal. Otherwise we compare against
    # a configurable similarity threshold.
    has_reliable_context = has_docs and (
        max_score is None or max_score >= MIN_RELEVANCE_SCORE_FOR_CONTEXT
    )

    llm = ChatGroq(
        model=model_name or OLLAMA_MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
    )

    if has_reliable_context:
        # RAG path: we have reasonably relevant documents, so we build context
        # and answer purely based on that context (no citations in the text).
        context_text = _format_docs(docs)
        prompt = get_rag_prompt_template()
        chain = prompt | llm | StrOutputParser()
        inputs = {
            "question": question,
            "chat_history": chat_history_text,
            "context": context_text,
        }
        answer: str = chain.invoke(inputs)
        source_docs: List[Document] = docs
    else:
        # Fallback path: either no docs or similarity scores are too low.
        # Here we let the model answer as a general-purpose assistant
        # (small-talk, open-domain questions, etc.), without forcing it
        # to use document context.
        prompt = get_general_chat_prompt_template()
        chain = prompt | llm | StrOutputParser()
        inputs = {
            "question": question,
            "chat_history": chat_history_text,
        }
        answer = chain.invoke(inputs)
        source_docs = []

    return {"answer": answer, "source_documents": source_docs}

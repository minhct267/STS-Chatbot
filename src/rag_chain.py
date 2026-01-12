from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from .config import (DEFAULT_TEMPERATURE, GROQ_MODEL_NAME, MAX_CONTEXT_CHUNKS,
                     MIN_RELEVANCE_SCORE_FOR_CONTEXT, OLLAMA_MODEL_NAME)
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


def _create_llm(
    provider: str | None,
    model_name: str | None,
    temperature: float | None,
) -> Any:
    temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    provider_normalised = (provider or "groq").strip().lower()
    if provider_normalised == "ollama":
        return ChatOllama(
            model=model_name or OLLAMA_MODEL_NAME,
            temperature=temp,
        )
    return ChatGroq(
        model=model_name or GROQ_MODEL_NAME,
        temperature=temp,
    )


def get_rag_chain(
    use_reranker: bool = True,
    model_name: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> Any:
    prompt = get_rag_prompt_template()
    llm = _create_llm(provider, model_name, temperature)

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
    provider: str | None = None,
    chat_history: str | None = None,
) -> Dict[str, Any]:
    chat_history_text = chat_history or ""

    docs, scores = retrieve_documents_with_scores(
        question,
        use_reranker=use_reranker,
    )
    max_score = max(scores) if scores else None

    has_docs = bool(docs)
    has_reliable_context = has_docs and (
        max_score is None or max_score >= MIN_RELEVANCE_SCORE_FOR_CONTEXT
    )

    llm = _create_llm(provider, model_name, temperature)

    if has_reliable_context:
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
        prompt = get_general_chat_prompt_template()
        chain = prompt | llm | StrOutputParser()
        inputs = {
            "question": question,
            "chat_history": chat_history_text,
        }
        answer = chain.invoke(inputs)
        source_docs = []

    return {"answer": answer, "source_documents": source_docs}

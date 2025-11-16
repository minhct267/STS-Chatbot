from __future__ import annotations

from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from .config import DEFAULT_TEMPERATURE, MAX_CONTEXT_CHUNKS, OLLAMA_MODEL_NAME
from .prompt import get_rag_prompt_template
from .retriever import retrieve_documents


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
    def build_context(question: str) -> str:
        docs = retrieve_documents(question, use_reranker=use_reranker)
        return _format_docs(docs)

    context_chain = itemgetter("question") | RunnableLambda(build_context)

    prompt = get_rag_prompt_template()
    llm = ChatOllama(
        model=model_name or OLLAMA_MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
    )

    rag_chain = (
        {
            "context": context_chain,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
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
    rag_chain = get_rag_chain(
        use_reranker=use_reranker,
        model_name=model_name,
        temperature=temperature,
    )
    inputs = {
        "question": question,
        "chat_history": chat_history or "",
    }
    answer: str = rag_chain.invoke(inputs)
    source_docs: List[Document] = retrieve_documents(
        question,
        use_reranker=use_reranker,
    )

    return {
        "answer": answer,
        "source_documents": source_docs,
    }

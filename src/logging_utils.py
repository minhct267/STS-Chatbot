from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LOG_DIR

INTERACTION_LOG_PATH = LOG_DIR / "interactions.jsonl"
FEEDBACK_LOG_PATH = LOG_DIR / "feedback.jsonl"


@dataclass
class RetrievedDocLog:
    source: str
    page: Optional[int]
    content: str


@dataclass
class InteractionLogRecord:
    timestamp: str
    session_id: str
    question: str
    answer: str
    model_name: str
    temperature: float
    use_reranker: bool
    retriever_k: int
    max_context_chunks: int
    retrieved_docs: List[RetrievedDocLog] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackLogRecord:
    timestamp: str
    session_id: str
    question: str
    answer: str
    feedback: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(record: InteractionLogRecord) -> None:
    try:
        _ensure_log_dir()
        payload = asdict(record)
        with INTERACTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to write interaction log: {exc}")


def log_feedback(record: FeedbackLogRecord) -> None:
    try:
        _ensure_log_dir()
        payload = asdict(record)
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to write feedback log: {exc}")


def build_interaction_record(
    *,
    session_id: str,
    question: str,
    answer: str,
    model_name: str,
    temperature: float,
    use_reranker: bool,
    retriever_k: int,
    max_context_chunks: int,
    source_docs: List[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> InteractionLogRecord:
    ts = datetime.now(timezone.utc).isoformat()

    retrieved_logs: List[RetrievedDocLog] = []
    for doc in source_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        try:
            page_int: Optional[int] = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_int = None

        content = getattr(doc, "page_content", "") or ""
        retrieved_logs.append(
            RetrievedDocLog(source=source, page=page_int, content=content),
        )

    return InteractionLogRecord(
        timestamp=ts,
        session_id=session_id,
        question=question,
        answer=answer,
        model_name=model_name,
        temperature=temperature,
        use_reranker=use_reranker,
        retriever_k=retriever_k,
        max_context_chunks=max_context_chunks,
        retrieved_docs=retrieved_logs,
        extra=extra or {},
    )


def build_feedback_record(
    *,
    session_id: str,
    question: str,
    answer: str,
    feedback: str,
    extra: Optional[Dict[str, Any]] = None,
) -> FeedbackLogRecord:
    ts = datetime.now(timezone.utc).isoformat()
    return FeedbackLogRecord(
        timestamp=ts,
        session_id=session_id,
        question=question,
        answer=answer,
        feedback=feedback,
        extra=extra or {},
    )

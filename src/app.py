from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import streamlit as st

# Ensure project root is on sys.path when running `streamlit run src/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (  # type: ignore  # noqa: E402
    APP_TITLE,
    AVAILABLE_OLLAMA_MODELS,
    DEFAULT_TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
    MAX_HISTORY_MESSAGES,
    OLLAMA_MODEL_NAME,
    RETRIEVER_K,
)
from src.logging_utils import (  # type: ignore  # noqa: E402
    build_interaction_record,
    log_interaction,
)
from src.rag_chain import answer_question  # type: ignore  # noqa: E402


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, str]] = []
    if "settings" not in st.session_state:
        st.session_state["settings"] = {
            "model_name": OLLAMA_MODEL_NAME,
            "use_reranker": True,
            "temperature": DEFAULT_TEMPERATURE,
        }
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())


def append_message(role: str, content: str) -> None:
    """Append a message to session_state and trim history if it grows too large (protect memory)."""
    messages: List[Dict[str, str]] = st.session_state["messages"]
    messages.append({"role": role, "content": content})

    # Keep at most ~2x the number of messages used for history to avoid bloating session_state
    max_len = MAX_HISTORY_MESSAGES * 2
    if len(messages) > max_len:
        overflow = len(messages) - max_len
        del messages[0:overflow]


def get_chat_history_text() -> str:
    """Format the most recent messages as text to feed into the prompt (conversational context)."""
    messages: List[Dict[str, str]] = st.session_state["messages"]
    recent = messages[-MAX_HISTORY_MESSAGES:]
    lines: List[str] = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def render_sidebar() -> Dict[str, object]:
    """Sidebar for configuring model, reranker and temperature."""
    settings = st.session_state["settings"]

    with st.sidebar:
        st.subheader("Settings")

        # Ollama model
        try:
            default_index = AVAILABLE_OLLAMA_MODELS.index(settings["model_name"])
        except ValueError:
            default_index = 0
        model_name = st.selectbox(
            "Ollama model",
            AVAILABLE_OLLAMA_MODELS,
            index=default_index,
        )

        # Reranker
        use_reranker = st.checkbox(
            "Use BGE reranker (higher accuracy, slightly slower)",
            value=bool(settings.get("use_reranker", True)),
        )

        # Temperature
        temperature = st.slider(
            "Temperature (0 = very conservative, 1 = more creative)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.get("temperature", DEFAULT_TEMPERATURE)),
            step=0.05,
        )

        settings.update(
            {
                "model_name": model_name,
                "use_reranker": use_reranker,
                "temperature": temperature,
            }
        )

    return settings


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
    st.title(APP_TITLE)

    init_session_state()
    settings = render_sidebar()

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents in the data/ folder..."):
        # Show user message
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history_text = get_chat_history_text()

        # Call RAG to answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating answer..."):
                result = answer_question(
                    prompt,
                    use_reranker=bool(settings["use_reranker"]),
                    model_name=str(settings["model_name"]),
                    temperature=float(settings["temperature"]),
                    chat_history=chat_history_text,
                )
                answer = result["answer"]
                source_docs = result["source_documents"]

                st.markdown(answer)

                if source_docs:
                    with st.expander("References (context used)"):
                        for idx, doc in enumerate(source_docs, start=1):
                            source = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page")
                            meta = f"[{idx}] {source}"
                            if page is not None:
                                meta += f" – page {page}"
                            st.markdown(f"**{meta}**")
                            st.markdown(doc.page_content)

        # Log interaction (does not affect UX if logging fails)
        try:
            record = build_interaction_record(
                session_id=st.session_state["session_id"],
                question=prompt,
                answer=answer,
                model_name=str(settings["model_name"]),
                temperature=float(settings["temperature"]),
                use_reranker=bool(settings["use_reranker"]),
                retriever_k=RETRIEVER_K,
                max_context_chunks=MAX_CONTEXT_CHUNKS,
                source_docs=source_docs,
                extra={
                    "app_version": "mvp2",
                },
            )
            log_interaction(record)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to log interaction in app: {exc}")

        append_message("assistant", answer)


if __name__ == "__main__":
    main()



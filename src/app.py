from __future__ import annotations

import os
import sys
from pathlib import Path
import re
from typing import Dict, List
from uuid import uuid4

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (APP_TITLE, AVAILABLE_GROQ_MODELS,
                        AVAILABLE_OLLAMA_MODELS, DEFAULT_LLM_PROVIDER,
                        DEFAULT_TEMPERATURE, MAX_CONTEXT_CHUNKS,
                        MAX_HISTORY_MESSAGES, GROQ_MODEL_NAME,
                        OLLAMA_MODEL_NAME, RETRIEVER_K)
from src.logging_utils import (build_feedback_record, build_interaction_record,
                               log_feedback, log_interaction)
from src.rag_chain import answer_question


def _init_groq_api_key_from_secrets() -> None:
    """Configure GROQ_API_KEY from Streamlit secrets when available.

    This allows deployments on Streamlit Community Cloud to manage the Groq
    API key via .streamlit/secrets.toml or the app settings UI, while keeping
    local development compatible with environment variables or .env files.
    """
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        # st.secrets may not be available in some non-Streamlit contexts;
        # in that case we simply fall back to existing environment settings.
        api_key = None

    if api_key and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = str(api_key)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, str]] = []
    if "settings" not in st.session_state:
        default_provider = (DEFAULT_LLM_PROVIDER or "groq").strip().lower()
        default_model = GROQ_MODEL_NAME if default_provider == "groq" else OLLAMA_MODEL_NAME
        st.session_state["settings"] = {
            "provider": default_provider,
            "model_name": default_model,
            "use_reranker": False,
            "temperature": DEFAULT_TEMPERATURE,
        }
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    if "last_interaction" not in st.session_state:
        st.session_state["last_interaction"] = None


def append_message(role: str, content: str) -> None:
    messages: List[Dict[str, str]] = st.session_state["messages"]
    messages.append({"role": role, "content": content})

    max_len = MAX_HISTORY_MESSAGES * 2
    if len(messages) > max_len:
        overflow = len(messages) - max_len
        del messages[0:overflow]


def get_chat_history_text() -> str:
    messages: List[Dict[str, str]] = st.session_state["messages"]
    recent = messages[-MAX_HISTORY_MESSAGES:]
    lines: List[str] = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def render_sidebar() -> Dict[str, object]:
    settings = st.session_state["settings"]

    header_left, header_right = st.columns([8, 2])
    with header_right:
        try:
            container = st.popover("⚙️ Settings")
        except Exception:
            container = st.expander("⚙️ Settings", expanded=False)

    with container:
        # LLM provider
        provider_display = "Groq" if settings.get("provider", "groq") == "groq" else "Ollama"
        provider_display = st.selectbox("LLM provider", ["Groq", "Ollama"], index=0 if provider_display == "Groq" else 1)
        provider = "groq" if provider_display == "Groq" else "ollama"

        # Model list depends on provider
        if provider == "groq":
            options = AVAILABLE_GROQ_MODELS
            label = "Groq model"
        else:
            options = AVAILABLE_OLLAMA_MODELS
            label = "Ollama model"
        try:
            default_index = options.index(settings.get("model_name", options[0]))
        except ValueError:
            default_index = 0
        model_name = st.selectbox(label, options, index=default_index)

        # Reranker - tickbox
        use_reranker = st.checkbox(
            "BGE reranker",
            value=bool(settings.get("use_reranker", False)),
            help="Higher accuracy, slightly slower",
        )

        # Temperature - sliding
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.get("temperature", DEFAULT_TEMPERATURE)),
            step=0.05,
            help="0 = conservative, 1 = more creative",
        )

        settings.update(
            {
                "provider": provider,
                "model_name": model_name,
                "use_reranker": use_reranker,
                "temperature": temperature,
            }
        )

    return settings


def _strip_reasoning_text(text: str) -> str:
    """Remove explicit chain-of-thought / thinking traces from model outputs.

    This targets common patterns like <think>...</think> and fenced blocks
    ```think ... ``` or ```thinking ... ```. If no patterns are found, the
    original text is returned unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```(?:think|thinking)[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    # Trim leftover excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or text


def render_feedback_controls() -> None:
    last = st.session_state.get("last_interaction")
    if not last:
        return

    feedback = last.get("feedback")
    if feedback == "up":
        st.caption("✅ Feedback recorded: marked as helpful.")
        return
    if feedback == "down":
        st.caption("✅ Feedback recorded: marked as not helpful.")
        return

    spacer, col1, col2 = st.columns([20, 1, 1])
    with col1:
        if st.button("👍", key="feedback_up"):
            _handle_feedback("up", last)
    with col2:
        if st.button("👎", key="feedback_down"):
            _handle_feedback("down", last)


def _handle_feedback(feedback_value: str, last: Dict[str, object]) -> None:
    try:
        record = build_feedback_record(
            session_id=st.session_state["session_id"],
            question=str(last["question"]),
            answer=str(last["answer"]),
            feedback=feedback_value,
            extra={"app_version": "mvp2"},
        )
        log_feedback(record)
        last["feedback"] = feedback_value
        st.session_state["last_interaction"] = last
    except Exception as exc:
        print(f"[WARN] Failed to log feedback: {exc}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide", initial_sidebar_state="collapsed")
    st.title(APP_TITLE)

    # Ensure Groq API key is available to ChatGroq instances.
    _init_groq_api_key_from_secrets()

    init_session_state()
    settings = render_sidebar()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything"):
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history_text = get_chat_history_text()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(
                    prompt,
                    use_reranker=bool(settings["use_reranker"]),
                    provider=str(settings.get("provider", "groq")),
                    model_name=str(settings["model_name"]),
                    temperature=float(settings["temperature"]),
                    chat_history=chat_history_text,
                )
                answer = _strip_reasoning_text(result["answer"])
                source_docs = result["source_documents"]

                st.markdown(answer)

                # Hide references panel per UI feedback (still logged for eval)

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
        except Exception as exc:
            print(f"[WARN] Failed to log interaction in app: {exc}")

        append_message("assistant", answer)

        st.session_state["last_interaction"] = {
            "question": prompt,
            "answer": answer,
            "feedback": None,
        }

    render_feedback_controls()


if __name__ == "__main__":
    main()

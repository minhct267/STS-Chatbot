## STS-Chatbot

RAG chatbot implemented fully in Python, using LangChain + Chroma + HuggingFace embeddings + Ollama and a Streamlit UI.

### Project structure

```text
STS-Chatbot/
├── data/                 # Source documents (PDF, DOCX, ...) used as knowledge base
├── chroma_db/            # Chroma database directory (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py         # Global config (paths, model names, RAG parameters)
│   ├── ingest.py         # Ingestion pipeline: read files, chunk, embed, save to Chroma
│   ├── vector_store.py   # Helpers to create / load Chroma vector store
│   ├── retriever.py      # Advanced retriever + reranker (BGE)
│   ├── rag_chain.py      # RAG chain with LangChain + Ollama
│   ├── logging_utils.py  # Structured interaction logging
│   ├── eval/             # Offline self-eval scripts
│   ├── app.py            # Streamlit app (chat UI)
│   └── utils.py          # Common utilities (logging, timing, ...)
├── requirements.txt      # Package list (reference; install into sts-chatbot env)
└── README.md
```

### Run the project

1. Activate the conda env `sts-chatbot` and make sure all dependencies are installed (see `requirements.txt`).
2. Put your PDF files into the `data/` directory.
3. Run ingestion to build the vector store:

```bash
python -m src.ingest
```

4. Start the Streamlit UI:

```bash
streamlit run src/app.py
```

### Notes

- All models (LLM and embeddings) run locally via Ollama and HuggingFace to minimise paid API usage.
- LangChain is used as the baseline RAG framework; LangGraph can be integrated later for self-evaluation and self-improvement workflows.
- By default, embeddings and the reranker run on GPU (`device="cuda"`). You can force CPU by setting env vars
  `STS_CHATBOT_EMBEDDING_DEVICE=cpu` and/or `STS_CHATBOT_RERANKER_DEVICE=cpu` before running the app.



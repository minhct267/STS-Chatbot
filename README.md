## STS-Chatbot

Self-Learning RAG Chatbot

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

Made with ❤️ by Mimo

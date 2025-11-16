import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory containing raw data (PDF, DOCX, ...)
DATA_DIR = BASE_DIR / "data"

# Directory for Chroma database (persist_directory)
CHROMA_DIR = BASE_DIR / "chroma_db"

# Directory for logs (interactions, eval, ...)
LOG_DIR = BASE_DIR / "logs"

# Collection name in Chroma
CHROMA_COLLECTION_NAME = "sts_chatbot_docs"

# HuggingFace embedding model (BGE, multilingual, good for search)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# LLM model name served via Ollama (can be changed here or via env var).
# Default is gemma3:12b for better quality; on weaker machines set STS_CHATBOT_OLLAMA_MODEL=gemma3:4b.
OLLAMA_MODEL_NAME = os.environ.get("STS_CHATBOT_OLLAMA_MODEL", "gemma3:12b")
AVAILABLE_OLLAMA_MODELS = ["gemma3:12b", "gemma3:4b"]


# Chunking parameters
CHUNK_SIZE = 1500  # characters
CHUNK_OVERLAP = 250  # characters


# Retrieval parameters
RETRIEVER_K = 20  # number of chunks retrieved from vector search
MAX_CONTEXT_CHUNKS = 8  # max chunks used to build context for the LLM


# Reranker (bge-reranker) – used to rerank chunks returned from Chroma
# Default uses the base model to save RAM; on stronger machines you can use "BAAI/bge-reranker-large".
USE_RERANKER = True
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Maximum number of chunks after rerank that will be used as context (default = MAX_CONTEXT_CHUNKS)
RERANKER_TOP_K = MAX_CONTEXT_CHUNKS

# Device to run the reranker on: "cpu" (safe on all machines) or "cuda" if you have a GPU
RERANKER_DEVICE = os.environ.get("STS_CHATBOT_RERANKER_DEVICE", "cpu")


# LLM parameters
DEFAULT_TEMPERATURE = 0.2


# Conversation memory (UI)
# Number of most recent messages (user + assistant) used to build chat history in the prompt
MAX_HISTORY_MESSAGES = 10


# Misc
APP_TITLE = "STS Chatbot - RAG (LangChain + Chroma + Ollama)"



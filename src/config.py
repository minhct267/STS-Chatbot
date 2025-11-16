import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory containing raw data
DATA_DIR = BASE_DIR / "data"

# Directory for Chroma database
CHROMA_DIR = BASE_DIR / "chroma_db"

# Directory for logs
LOG_DIR = BASE_DIR / "logs"

# Collection name in Chroma
CHROMA_COLLECTION_NAME = "sts_chatbot_docs"

# HuggingFace embedding model (BGE, multilingual, good for search)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# GPU if available
EMBEDDING_DEVICE = os.environ.get("STS_CHATBOT_EMBEDDING_DEVICE", "cuda")
RERANKER_DEVICE = os.environ.get("STS_CHATBOT_RERANKER_DEVICE", "cuda")

# LLM model served via Ollama
OLLAMA_MODEL_NAME = os.environ.get("STS_CHATBOT_OLLAMA_MODEL", "gemma3:12b")
AVAILABLE_OLLAMA_MODELS = ["gemma3:12b", "llama3.3:latest", "gpt-oss:120b", "gemma3:27b"]

# Chunking parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

# Ingestion parameters
MIN_PAGE_CHARS = 200
MIN_CHUNK_CHARS = 100

# Version tag for the current indexing configuration
INDEX_VERSION = "v1.0.0"

# Retrieval parameters
RETRIEVER_K = 20
MAX_CONTEXT_CHUNKS = 8

# Reranker
USE_RERANKER = True
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Maximum number of chunks after rerank
RERANKER_TOP_K = MAX_CONTEXT_CHUNKS

# LLM parameters
DEFAULT_TEMPERATURE = 0.2

# Conversation memory
MAX_HISTORY_MESSAGES = 10

# App title
APP_TITLE = "STS Chatbot"

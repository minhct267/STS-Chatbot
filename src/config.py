import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env file (if present).
# This allows you to define GROQ_API_KEY and other settings during development.
load_dotenv()

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

# LLM model served via Groq
# You can override this by setting STS_CHATBOT_GROQ_MODEL in the environment.
GROQ_MODEL_NAME = os.environ.get("STS_CHATBOT_GROQ_MODEL", "llama-3.1-8b-instant")

# Backwards-compatible aliases used elsewhere in the app (e.g. sidebar model picker).
OLLAMA_MODEL_NAME = GROQ_MODEL_NAME
AVAILABLE_OLLAMA_MODELS = [GROQ_MODEL_NAME]

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

# Minimum similarity/relevance score required to trust retrieved context.
# This is only used when the underlying vector store exposes relevance scores;
# if no scores are available, this threshold is ignored and the presence of
# any documents is treated as a positive retrieval signal.
MIN_RELEVANCE_SCORE_FOR_CONTEXT = 0.3

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

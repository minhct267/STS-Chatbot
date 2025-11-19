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

# LLM providers
AVAILABLE_LLM_PROVIDERS = ["groq", "ollama"]
DEFAULT_LLM_PROVIDER = os.environ.get("STS_CHATBOT_LLM_PROVIDER", "groq")

# Collection name in Chroma
CHROMA_COLLECTION_NAME = "sts_chatbot_docs"

# HuggingFace embedding model (BGE, multilingual, good for search)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Device selection (CUDA if available, else CPU), overridable via env vars
def _auto_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEFAULT_DEVICE = _auto_device()
EMBEDDING_DEVICE = os.environ.get("STS_CHATBOT_EMBEDDING_DEVICE", DEFAULT_DEVICE)
RERANKER_DEVICE = os.environ.get("STS_CHATBOT_RERANKER_DEVICE", DEFAULT_DEVICE)

# LLM model served via Groq
# You can override this by setting STS_CHATBOT_GROQ_MODEL in the environment.
GROQ_MODEL_NAME = os.environ.get("STS_CHATBOT_GROQ_MODEL", "llama-3.1-8b-instant")

# Available Groq models
AVAILABLE_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-prompt-guard-2-86m",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

# Ollama models (used when switching provider to Ollama)
OLLAMA_MODEL_NAME = os.environ.get("STS_CHATBOT_OLLAMA_MODEL", "llama3.1:8b")
AVAILABLE_OLLAMA_MODELS = [
    "llama3.1:8b",
    "gemma3:4b",
    "llava:13b",
    "qwen3:32b",
    "gpt-oss:120b",
]

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
USE_RERANKER = False
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Maximum number of chunks after rerank
RERANKER_TOP_K = MAX_CONTEXT_CHUNKS

# LLM parameters
DEFAULT_TEMPERATURE = 0.2

# Conversation memory
MAX_HISTORY_MESSAGES = 10

# App title
APP_TITLE = "STS Chatbot"

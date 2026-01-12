import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Chroma database directory
CHROMA_DIR = BASE_DIR / "chroma_db"

# Logs directory
LOG_DIR = BASE_DIR / "logs"

# Available LLM providers
AVAILABLE_LLM_PROVIDERS = ["groq", "ollama"]
DEFAULT_LLM_PROVIDER = os.environ.get("STS_CHATBOT_LLM_PROVIDER", "groq")

# Collection name in Chroma
CHROMA_COLLECTION_NAME = "sts_chatbot_docs"

# HuggingFace embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Device selection
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DEVICE = os.environ.get("STS_CHATBOT_EMBEDDING_DEVICE", DEFAULT_DEVICE)
RERANKER_DEVICE = os.environ.get("STS_CHATBOT_RERANKER_DEVICE", DEFAULT_DEVICE)

# LLM model served via Groq
GROQ_MODEL_NAME = os.environ.get("STS_CHATBOT_GROQ_MODEL", "llama-3.1-8b-instant")

# Groq models
AVAILABLE_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-prompt-guard-2-86m",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

# Ollama models
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

# Index version
INDEX_VERSION = "v1.0.0"

# Retrieval parameters
RETRIEVER_K = 20
MAX_CONTEXT_CHUNKS = 8

# Minimum similarity/relevance score required to trust retrieved context
MIN_RELEVANCE_SCORE_FOR_CONTEXT = 0.3

# Reranker
USE_RERANKER = False
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Maximum number of chunks after rerank
RERANKER_TOP_K = MAX_CONTEXT_CHUNKS

# LLM temperature
DEFAULT_TEMPERATURE = 0.2

# Maximum number of messages in chat history
MAX_HISTORY_MESSAGES = 10

# App title
APP_TITLE = "STS Chatbot"

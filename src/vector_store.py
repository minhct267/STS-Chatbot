from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL_NAME


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialise HuggingFaceEmbeddings with the BGE-m3 model.

    The model is loaded from HuggingFace Hub (free) and runs locally via
    sentence-transformers + torch.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # switch to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(
    persist_directory: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Load Chroma vector store persisted on disk."""
    persist_directory = persist_directory or CHROMA_DIR
    collection_name = collection_name or CHROMA_COLLECTION_NAME

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma directory does not exist yet: {persist_directory}. "
            "Run ingest first (python -m src.ingest)."
        )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def rebuild_vector_store_from_documents(
    documents: List[Document],
    persist_directory: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Rebuild the entire vector store from a list of documents.

    This function deletes the existing Chroma directory (if any) and builds
    a fresh index from the given documents.
    """
    persist_directory = persist_directory or CHROMA_DIR
    collection_name = collection_name or CHROMA_COLLECTION_NAME

    persist_directory.mkdir(parents=True, exist_ok=True)

    # Remove old data (if any) by deleting the whole directory.
    # Chroma will recreate the required structure.
    for child in persist_directory.iterdir():
        if child.is_file():
            child.unlink()
        else:
            # Recursively delete subdirectories
            for sub in child.rglob("*"):
                if sub.is_file():
                    sub.unlink()
            child.rmdir()

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    return vectorstore



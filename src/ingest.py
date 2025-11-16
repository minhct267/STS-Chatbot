from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from .utils import time_block
from .vector_store import rebuild_vector_store_from_documents


def load_pdf_documents(data_dir: Path) -> List[Document]:
    """Load all PDF files in the data directory into a list of Documents."""
    docs: List[Document] = []
    pdf_paths = sorted(data_dir.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in directory: {data_dir}")
        return docs

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        file_docs = loader.load()
        for d in file_docs:
            d.metadata.setdefault("source", str(pdf_path.name))
        docs.extend(file_docs)

    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks while keeping enough context for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
    )
    return splitter.split_documents(documents)


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")

    with time_block("Load PDFs from data/"):
        raw_docs = load_pdf_documents(DATA_DIR)

    if not raw_docs:
        print("No documents to index. Exiting.")
        return

    with time_block("Chunk documents"):
        chunked_docs = chunk_documents(raw_docs)
        print(f"Total chunks after splitting: {len(chunked_docs)}")

    with time_block("Build Chroma vector store"):
        vectorstore = rebuild_vector_store_from_documents(chunked_docs)
        print(
            "Ingestion completed. Vector store persisted. "
            f"Number of documents in collection: {vectorstore._collection.count()}"  # type: ignore[attr-defined]
        )


if __name__ == "__main__":
    main()



from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, INDEX_VERSION,
                     LOG_DIR, MIN_CHUNK_CHARS, MIN_PAGE_CHARS)
from .utils import time_block
from .vector_store import rebuild_vector_store_from_documents

SENTENCE_END_CHARS: Set[str] = {".", "?", "!", "…", ":", ";", "。", "？", "！"}


def _strip_non_printable(text: str) -> str:
    return "".join(ch if ch.isprintable() or ch in {"\n", "\t"} else " " for ch in text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def _fix_broken_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    paragraphs: List[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            paragraphs.append(buffer.strip())
        buffer = ""

    for line in lines:
        if not line:
            flush_buffer()
            continue

        if not buffer:
            buffer = line
            continue

        last_char = buffer[-1]
        is_sentence_end = last_char in SENTENCE_END_CHARS
        starts_with_bullet = line.startswith(("-", "•", "*"))

        if not is_sentence_end and not starts_with_bullet:
            buffer = f"{buffer} {line}"
        else:
            flush_buffer()
            buffer = line

    flush_buffer()

    merged = "\n\n".join(paragraphs)
    merged = re.sub(r"[ \t]+", " ", merged)
    return merged.strip()


def _compute_repeated_edge_lines(pages: Sequence[str], *, top_k: int = 1, threshold: float = 0.6) -> Dict[str, Set[str]]:
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    for text in pages:
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not raw_lines:
            continue
        header_candidates = raw_lines[:top_k]
        footer_candidates = raw_lines[-top_k:]
        for ln in header_candidates:
            header_counts[ln] = header_counts.get(ln, 0) + 1
        for ln in footer_candidates:
            footer_counts[ln] = footer_counts.get(ln, 0) + 1

    min_count = max(1, int(len(pages) * threshold))
    header_set = {ln for ln, c in header_counts.items() if c >= min_count}
    footer_set = {ln for ln, c in footer_counts.items() if c >= min_count}

    return {"header": header_set, "footer": footer_set}


def _remove_headers_footers(text: str, headers: Set[str], footers: Set[str]) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and lines[0].strip() in headers:
        lines.pop(0)
    while lines and lines[-1].strip() in footers:
        lines.pop()
    return "\n".join(lines)


def _extract_section_title(text: str, max_len: int = 120) -> str | None:
    for line in text.splitlines():
        title = line.strip()
        if not title:
            continue
        if len(title) > max_len:
            return title[: max_len - 3] + "..."
        return title
    return None


def _preprocess_pdf_pages(file_docs: List[Document], *, pdf_name: str) -> List[Document]:
    if not file_docs:
        return []

    raw_pages = [doc.page_content or "" for doc in file_docs]
    hf_sets = _compute_repeated_edge_lines(raw_pages)
    headers = hf_sets["header"]
    footers = hf_sets["footer"]

    cleaned_docs: List[Document] = []

    for doc in file_docs:
        raw_text = doc.page_content or ""
        text = _remove_headers_footers(raw_text, headers, footers)
        text = _strip_non_printable(text)
        text = _normalize_whitespace(text)
        text = _fix_broken_lines(text)

        non_ws_len = len(re.sub(r"\s+", "", text))
        if non_ws_len < MIN_PAGE_CHARS:
            continue

        doc.page_content = text

        metadata = doc.metadata or {}
        metadata.setdefault("source", pdf_name)
        metadata.setdefault("doc_id", Path(pdf_name).stem)
        metadata["index_version"] = INDEX_VERSION

        section_title = _extract_section_title(text)
        if section_title:
            metadata.setdefault("section_title", section_title)

        doc.metadata = metadata
        cleaned_docs.append(doc)

    return cleaned_docs


def load_pdf_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    pdf_paths = sorted(data_dir.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in directory: {data_dir}")
        return docs

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        file_docs = loader.load()
        cleaned = _preprocess_pdf_pages(file_docs, pdf_name=str(pdf_path.name))
        docs.extend(cleaned)

    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n- ",
            "\n• ",
            "\n* ",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
        ],
    )
    return splitter.split_documents(documents)


def _filter_and_deduplicate_chunks(chunks: List[Document]) -> List[Document]:

    def normalise_for_key(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    unique_chunks: List[Document] = []
    seen_keys: Set[str] = set()

    for doc in chunks:
        content = doc.page_content or ""
        non_ws_len = len(re.sub(r"\s+", "", content))
        if non_ws_len < MIN_CHUNK_CHARS:
            continue
        key = normalise_for_key(content)
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_chunks.append(doc)

    return unique_chunks


def _enrich_chunk_metadata(chunks: List[Document]) -> List[Document]:
    grouped: Dict[tuple, List[int]] = {}
    for idx, doc in enumerate(chunks):
        md = doc.metadata or {}
        key = (md.get("doc_id"), md.get("source"))
        grouped.setdefault(key, []).append(idx)

    for key, indices in grouped.items():
        num_chunks = len(indices)
        for local_idx, global_idx in enumerate(indices):
            doc = chunks[global_idx]
            md = doc.metadata or {}
            md.setdefault("doc_id", key[0])
            md.setdefault("source", key[1])
            md["chunk_index"] = local_idx
            md["num_chunks"] = num_chunks
            md.setdefault("index_version", INDEX_VERSION)
            doc.metadata = md

    return chunks


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

    with time_block("Filter / deduplicate chunks and enrich metadata"):
        processed_chunks = _filter_and_deduplicate_chunks(chunked_docs)
        processed_chunks = _enrich_chunk_metadata(processed_chunks)
        print(f"Total chunks after cleaning & deduplication: {len(processed_chunks)}")

    with time_block("Build Chroma vector store"):
        vectorstore = rebuild_vector_store_from_documents(processed_chunks)
        print(
            "Ingestion completed. Vector store persisted. "
            f"Number of documents in collection: {vectorstore._collection.count()}"
        )


if __name__ == "__main__":
    main()

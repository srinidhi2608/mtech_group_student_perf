"""
PDF ingestion pipeline using PyMuPDF to extract text and chunk documents.
Provides a backwards-compatible wrapper `ingest_pdf_to_chunks` expected by app/main.py.
"""
import os
import logging
from typing import List, Dict
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks (characters).
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# primary ingest function (kept for compatibility)
def ingest_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Ingest a PDF file and return chunked documents with metadata.
    Returned documents: list of dicts with keys: text, source, chunk_id, metadata
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    logger.info(f"Ingesting PDF: {pdf_path}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}")
        return []

    # Chunk text
    chunks = chunk_text(text, chunk_size, overlap)
    logger.info(f"Created {len(chunks)} chunks from {pdf_path}")

    # Add metadata
    filename = os.path.basename(pdf_path)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "source": filename,
            "chunk_id": i,
            "metadata": {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })

    return documents


def batch_ingest_pdfs(pdf_paths: List[str], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Ingest multiple PDF files.
    """
    all_documents = []
    for pdf_path in pdf_paths:
        docs = ingest_pdf(pdf_path, chunk_size, overlap)
        all_documents.extend(docs)

    logger.info(f"Ingested {len(pdf_paths)} PDFs, total {len(all_documents)} chunks")
    return all_documents


# Backwards-compatible wrapper expected by app/main.py
def ingest_pdf_to_chunks(pdf_path: str, metadata: dict = None, chunk_size: int = 500, overlap: int = 50):
    """
    Compatibility wrapper: returns list of chunks in the shape app expects:
      [{"id": <str_or_int>, "text": <chunk_text>, "metadata": {...}}, ...]
    - If caller provides `metadata`, it will be merged into each chunk's metadata (caller keys override).
    """
    docs = ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    out = []
    for d in docs:
        meta = d.get('metadata', {}).copy()
        # preserve original source unless caller overrides
        meta.setdefault('source', d.get('source'))
        if metadata:
            # caller metadata overrides/extends
            meta.update(metadata)
        out.append({
            "id": str(d.get('chunk_id')),
            "text": d.get("text", ""),
            "metadata": meta
        })
    return out
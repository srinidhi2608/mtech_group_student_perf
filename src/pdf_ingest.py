"""
PDF ingestion pipeline using PyMuPDF to extract text and chunk documents.
"""
import os
import logging
from typing import List, Dict
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
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
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
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

def ingest_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Ingest a PDF file and return chunked documents with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of dictionaries containing chunk text and metadata
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
    
    Args:
        pdf_paths: List of paths to PDF files
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of all document chunks from all PDFs
    """
    all_documents = []
    for pdf_path in pdf_paths:
        docs = ingest_pdf(pdf_path, chunk_size, overlap)
        all_documents.extend(docs)
    
    logger.info(f"Ingested {len(pdf_paths)} PDFs, total {len(all_documents)} chunks")
    return all_documents

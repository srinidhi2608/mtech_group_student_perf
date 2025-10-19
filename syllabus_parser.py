import pdfplumber
import re
from typing import Dict

def parse_pdf_to_text(filelike) -> str:
    """
    Extracts text from an uploaded PDF (file-like object) using pdfplumber.
    Returns a cleaned text string.
    """
    text_parts = []
    with pdfplumber.open(filelike) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
    text_full = "\n".join(text_parts)
    # simple cleaning
    text_full = re.sub(r'\s+', ' ', text_full)
    return text_full

def build_keyword_index(syllabus_texts: Dict[str, str], keywords: list):
    """
    For a set of keywords, returns a dict mapping filename -> matched keywords found
    """
    res = {}
    for name, text in syllabus_texts.items():
        found = set()
        lower = text.lower()
        for kw in keywords:
            if kw.lower() in lower:
                found.add(kw)
        res[name] = list(found)
    return res
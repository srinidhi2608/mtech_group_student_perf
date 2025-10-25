from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json
from typing import List, Optional

# Reuse helper modules (these should exist in src/)
# - src/pdf_ingest.py -> ingest_pdf_to_chunks
# - src/embeddings.py -> EmbeddingService
# - src/vectorstore.py -> FaissStore
# - src/company_matcher.py -> company matching helpers
# - src/recommender.py -> course recommender helpers (optional)
try:
    from src.pdf_ingest import ingest_pdf_to_chunks
except Exception:
    ingest_pdf_to_chunks = None

try:
    from src.embeddings import EmbeddingService
except Exception:
    EmbeddingService = None

try:
    from src.vectorstore import FaissStore
except Exception:
    FaissStore = None

try:
    from src.company_matcher import load_companies, build_company_skill_profiles, match_student_courses_to_companies
except Exception:
    load_companies = None
    build_company_skill_profiles = None
    match_student_courses_to_companies = None

try:
    from src.recommender import load_courses_catalog, recommend_courses_for_company
except Exception:
    load_courses_catalog = None
    recommend_courses_for_company = None

app = FastAPI(title="Student Perf & Chat API")

# Artifact paths
SEM6_MODEL_PATH = "models/sem6_pred.joblib"
FAISS_INDEX_PATH = "models/faiss.index"
FAISS_META_PATH = "models/faiss_meta.json"
UPLOAD_DIR = "uploads"
COURSES_CSV = "data/courses_catalog.csv"
COMPANIES_CSV = "Glassdoor_Salary_Cleaned_Version.csv"

# Load sem6 model artifact (if present)
sem6_model = None
sem6_mlb = None
sem6_feature_cols = None
if os.path.exists(SEM6_MODEL_PATH):
    try:
        art = joblib.load(SEM6_MODEL_PATH)
        sem6_model = art.get("model")
        sem6_mlb = art.get("mlb")
        sem6_feature_cols = art.get("feature_cols")
    except Exception:
        sem6_model = None
        sem6_mlb = None
        sem6_feature_cols = None

# Initialize embedding service and vectorstore (if available)
embedder = None
vectorstore = None
if EmbeddingService is not None:
    try:
        embedder = EmbeddingService(model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    except Exception:
        embedder = None

if FaissStore is not None and embedder is not None:
    try:
        dim = embedder.embed_text("hello").shape[0]
        vectorstore = FaissStore(dim=dim, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
        # load persisted index/meta if present
        vectorstore.load()
    except Exception:
        vectorstore = None

# Load companies & build profiles (non-fatal)
companies_df = pd.DataFrame()
company_profiles = {}
if load_companies is not None and build_company_skill_profiles is not None and os.path.exists(COMPANIES_CSV):
    try:
        companies_df = load_companies(COMPANIES_CSV)
        company_profiles, _ = build_company_skill_profiles(companies_df, text_col="Job Description")
    except Exception:
        companies_df = pd.DataFrame()
        company_profiles = {}

# Load courses catalog (non-fatal)
courses_df = pd.DataFrame()
if load_courses_catalog is not None and os.path.exists(COURSES_CSV):
    try:
        courses_df = load_courses_catalog(COURSES_CSV)
    except Exception:
        courses_df = pd.DataFrame()

# ---- Request/response models ----
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    use_llm: Optional[bool] = False  # if True and OPENAI_API_KEY set, attempt LLM synthesis

class UploadResponse(BaseModel):
    status: str
    chunks_added: int
    source: str

# ---- Utility helpers ----
def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Endpoints ----
@app.get("/")
def health():
    return {
        "status": "ok",
        "sem6_model": sem6_model is not None,
        "embedder": embedder is not None,
        "vectorstore": vectorstore is not None,
        "companies_indexed": len(company_profiles) > 0,
        "courses_loaded": not courses_df.empty
    }

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), title: Optional[str] = Form(None), description: Optional[str] = Form(None)):
    """
    Upload a PDF file, extract text, chunk, embed and add to vectorstore.
    """
    if ingest_pdf_to_chunks is None or embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="PDF ingestion/embedding/vectorstore not available. Ensure src/pdf_ingest, src/embeddings, src/vectorstore exist and are importable.")
    ensure_upload_dir()
    filename = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    # save file
    try:
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # ingest into chunks
    try:
        chunks = ingest_pdf_to_chunks(path, metadata={"source": filename, "title": title or filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {e}")

    texts = [c["text"] for c in chunks]
    if not texts:
        return {"status": "ok", "chunks_added": 0, "source": filename}

    # embed and add to vectorstore
    try:
        vectors = embedder.embed_texts(texts)
        metadatas = []
        for idx, c in enumerate(chunks):
            meta = {
                "source": c.get("metadata", {}).get("source", filename),
                "title": c.get("metadata", {}).get("title", title or filename),
                "text": c["text"],
                "chunk_id": c.get("id", idx)
            }
            metadatas.append(meta)
        vectorstore.add(vectors.astype("float32"), metadatas)
        vectorstore.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed/index chunks: {e}")

    return {"status": "ok", "chunks_added": len(texts), "source": filename}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Query the vectorstore for top-k matches and optionally synthesize an answer using an LLM (if OPENAI_API_KEY set).
    Returns retrieved snippets and, if available, a generated answer.
    """
    if embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="Embeddings or vectorstore not available on the server.")

    q = req.query
    k = int(req.top_k or 8)
    q_vec = embedder.embed_text(q).astype("float32")
    # search
    try:
        results = vectorstore.search(q_vec, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # build simple response
    retrieved = []
    for r in results:
        retrieved.append({
            "source": r.get("source"),
            "title": r.get("title"),
            "text": (r.get("text")[:800] + "...") if r.get("text") and len(r.get("text")) > 800 else r.get("text"),
            "score": r.get("score")
        })

    # optional LLM synthesis using OpenAI (if configured)
    answer = None
    if req.use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            system = "You are a helpful assistant. Use the provided context snippets to answer the user query concisely."
            context_text = "\n\n---\n\n".join([f"Source: {r['source']}\n\n{r['text']}" for r in retrieved])
            prompt = f"User query: {q}\n\nContext:\n{context_text}\n\nAnswer:"
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception:
            answer = None

    return {"query": q, "answer": answer, "retrieved": retrieved}

# Small search helper endpoint
@app.get("/search")
def search(q: str, k: int = 6):
    if embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="Embeddings or vectorstore not available.")
    q_vec = embedder.embed_text(q).astype("float32")
    results = vectorstore.search(q_vec, k=k)
    return {"query": q, "results": results}
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List, Optional
from openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


# helper modules we added above
from src.pdf_ingest import ingest_pdf_to_chunks
from src.embeddings import EmbeddingService, get_embeddings_model
from src.vectorstore import FaissStore
from src.company_matcher import load_companies, build_company_skill_profiles, match_student_to_companies_by_flags, match_student_to_companies_by_text
from src.recommender import load_courses_catalog, recommend_courses_for_skills

app = FastAPI(title="Student (RAG + Sem6 predictor)")

# Artifact paths and resources
SEM6_MODEL_PATH = "models/sem6_pred.joblib"
COMPANIES_CSV = "Glassdoor_Salary_Cleaned_Version.csv"
COURSES_CSV = "data/courses_catalog.csv"
UPLOAD_DIR = "uploads"
FAISS_INDEX_PATH = "models/faiss.index"
FAISS_META_PATH = "models/faiss_meta.json"

# Load sem6 artifact
sem6_model = sem6_mlb = sem6_feature_cols = None
if os.path.exists(SEM6_MODEL_PATH):
    try:
        art = joblib.load(SEM6_MODEL_PATH)
        sem6_model = art.get("model")
        sem6_mlb = art.get("mlb")
        sem6_feature_cols = art.get("feature_cols")
    except Exception:
        sem6_model = sem6_mlb = sem6_feature_cols = None

# Load companies and profiles
companies_df = pd.DataFrame()
company_profiles = {}
skill_flag_columns = ['python_yn','R_yn','spark','aws','excel']  # as per your dataset
if os.path.exists(COMPANIES_CSV):
    companies_df = load_companies(COMPANIES_CSV)
    # if explicit flags exist, prefer them; also build text profiles as fallback
    company_profiles_text, tfidf_vect = build_company_skill_profiles(companies_df, text_col='Job Description')

# Load courses
courses_df = pd.DataFrame()
if os.path.exists(COURSES_CSV):
    courses_df = load_courses_catalog(COURSES_CSV)

# Initialize embeddings & vectorstore
embedder = None
vectorstore = None
try:
    embedder = get_embeddings_model(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    dim = embedder.get_embedding_dim()
    vectorstore = FaissStore(dim=dim, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH)
    # if persisted index exists, FaissStore.load() will be called in constructor
    print("EMBED_MODEL and Faiss model are loaded")
except Exception:
    embedder = None
    vectorstore = None

def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

class StudentPayload(BaseModel):
    PUC_Score: float
    Sem_1: float
    Sem_2: float
    Sem_3: float
    Sem_4: float
    Sem_5: float
    Attendance_percent: float
    Study_Hours_Per_Day: float
    Sleep_Hours_Per_Day: float
    Courses_Completed: Optional[str] = ""

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    if ingest_pdf_to_chunks is None or embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="PDF ingestion/embedding/vectorstore not available.")
    ensure_upload_dir()
    filename = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)
    chunks = ingest_pdf_to_chunks(path, metadata={"source": filename, "title": title or filename})
    if not chunks:
        return {"status": "ok", "chunks_added": 0, "source": filename}
    texts = [c['text'] for c in chunks]
    vectors = embedder.embed_texts(texts).astype("float32")
    metadatas = []
    for i,c in enumerate(chunks):
        meta = {"source": filename, "title": title or filename, "text": c['text'], "chunk_id": c.get('id', i)}
        metadatas.append(meta)
    vectorstore.add(vectors, metadatas)
    vectorstore.save()
    return {"status": "ok", "chunks_added": len(texts), "source": filename}

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    use_llm: Optional[bool] = True

@app.post("/chat")
def chat(req: ChatRequest):

    if embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="Embeddings or vectorstore not available.")
    q = req.query
    qvec = embedder.embed_text(q).astype("float32")
    results = vectorstore.search(qvec, k=req.top_k or 8)
    # results is list of metadata dicts (with score)
    # Build context text
    context = "\n\n---\n\n".join([r.get('text','')[:1500] for r in results])
    answer = None
    print(os.getenv("OPENAI_API_KEY"))
    if req.use_llm :
        try:

            print("Load the model for LLM")
            llm = ChatOllama(model="myname_model:latest")
            response = llm.invoke(q)
            print(response.content)
            answer = response.content

        except Exception as e:
            print("something went wrong", e)
            answer = None
    return {"query": q, "answer": answer, "retrieved": results}

@app.post("/student_predict")
def student_predict(payload: StudentPayload):
    if sem6_model is None or sem6_mlb is None or sem6_feature_cols is None:
        raise HTTPException(status_code=500, detail="Sem_6 model not available. Train first.")
    data = payload.dict()
    courses = [c.strip().lower() for c in data.get('Courses_Completed','').split(',') if c.strip()]
    try:
        courses_vec = sem6_mlb.transform([courses])
    except Exception:
        import numpy as np
        courses_vec = np.zeros((1, len(sem6_mlb.classes_)))
    courses_df_local = pd.DataFrame(courses_vec, columns=[f"course_{c}" for c in sem6_mlb.classes_])
    feature_vals = pd.DataFrame([{
        "PUC_Score": data["PUC_Score"],
        "Sem_1": data["Sem_1"],
        "Sem_2": data["Sem_2"],
        "Sem_3": data["Sem_3"],
        "Sem_4": data["Sem_4"],
        "Sem_5": data["Sem_5"],
        "Attendance_percent": data["Attendance_percent"],
        "Study_Hours_Per_Day": data["Study_Hours_Per_Day"],
        "Sleep_Hours_Per_Day": data["Sleep_Hours_Per_Day"],
    }])
    X = pd.concat([feature_vals.reset_index(drop=True), courses_df_local.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=sem6_feature_cols).fillna(0)
    pred = float(sem6_model.predict(X)[0])
    return {"predicted_Sem_6": pred}

@app.post("/student_recommendations")
def student_recommendations(payload: StudentPayload, top_n: Optional[int] = 5):
    # 1) predict Sem_6
    pred_resp = student_predict(payload)
    predicted = pred_resp.get("predicted_Sem_6")
    # 2) extract student skills from Courses_Completed
    courses = [c.strip().lower() for c in payload.Courses_Completed.split(',') if c.strip()]
    # 3) find companies matching skills: prefer explicit flags if present
    matches = []
    if 'python_yn' in companies_df.columns or 'R_yn' in companies_df.columns or 'spark' in companies_df.columns:
        matches = match_student_to_companies_by_flags(courses, companies_df, skill_flag_columns, top_n=top_n)
    else:
        matches = match_student_to_companies_by_text(courses, company_profiles_text, companies_df, top_n=top_n)
    # 4) determine missing/desired skills (simple heuristic: look at top company skill set union minus student skills)
    target_skills = set()
    for m in matches:
        # if company_profiles_text keyed by name exists
        cname = m['company']
        sks = company_profiles_text.get(cname, set())
        target_skills.update([s for s in sks if s not in courses])
    # 5) recommend courses to fill missing skills
    recs = []
    if not courses_df.empty and target_skills:
        recs = recommend_courses_for_skills(list(target_skills), courses_df, top_n=top_n)
    return {"predicted_Sem_6": predicted, "company_matches": matches, "recommended_courses": recs}
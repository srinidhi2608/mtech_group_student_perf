from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging
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

# Persistent memory imports (defensive)
sqlite_memory = None
semantic_memory = None
try:
    from src.persistent_memory import SQLiteMemory, SemanticMemory
    logging.info("Persistent memory modules imported successfully")
except Exception as e:
    logging.warning(f"Could not import persistent_memory module: {e}")
    SQLiteMemory = None
    SemanticMemory = None

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

# Initialize persistent memory modules
if SQLiteMemory is not None and embedder is not None:
    try:
        sqlite_memory = SQLiteMemory(db_path="models/chat_history.db")
        semantic_memory = SemanticMemory(
            embedder=embedder,
            index_path="models/semantic_memory.index",
            meta_path="models/semantic_memory.json"
        )
        logging.info("Persistent memory initialized successfully")
    except Exception as e:
        logging.warning(f"Failed to initialize persistent memory: {e}")
        sqlite_memory = None
        semantic_memory = None
else:
    logging.info("Persistent memory not available (missing dependencies)")


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
    student_id: Optional[str] = None

@app.post("/chat")
def chat(req: ChatRequest):

    if embedder is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="Embeddings or vectorstore not available.")
    
    q = req.query
    
    # Derive student_id (default to 'anonymous' for backward compatibility)
    student_id = req.student_id or 'anonymous'
    
    # Initialize response fields
    retrieved_qa = []
    chat_history = []
    
    # Retrieve recent chat history from SQLiteMemory
    if sqlite_memory is not None:
        try:
            chat_history = sqlite_memory.get_last_n(student_id, n=12)
        except Exception as e:
            logging.warning(f"Failed to retrieve chat history: {e}")
            chat_history = []
    
    # Retrieve student-specific QA from SemanticMemory
    if semantic_memory is not None:
        try:
            retrieved_qa = semantic_memory.retrieve_similar(student_id, q, k=3)
        except Exception as e:
            logging.warning(f"Failed to retrieve semantic memories: {e}")
            retrieved_qa = []
    
    # Perform existing vectorstore search for uploaded documents
    qvec = embedder.embed_text(q).astype("float32")
    results = vectorstore.search(qvec, k=req.top_k or 8)
    
    # Build enhanced context for the prompt
    # Start with system message
    prompt_parts = []
    
    # Add relevant prior student Q&A if available
    if retrieved_qa:
        qa_context = "Relevant prior student Q&A:\n"
        for i, qa in enumerate(retrieved_qa[:3], 1):
            qa_context += f"{i}. Q: {qa.get('text', '')[:200]}\n"
        prompt_parts.append(qa_context)
    
    # Add relevant documents from vectorstore
    if results:
        doc_context = "Relevant documents:\n"
        doc_context += "\n\n---\n\n".join([r.get('text','')[:1500] for r in results])
        prompt_parts.append(doc_context)
    
    # Add recent conversation context
    if chat_history:
        conv_context = "Recent conversation:\n"
        for msg in chat_history[-6:]:  # Last 6 messages (3 turns)
            role = msg.get('role', 'user')
            text = msg.get('text', '')[:200]
            conv_context += f"{role.capitalize()}: {text}\n"
        prompt_parts.append(conv_context)
    
    # Add current user question
    prompt_parts.append(f"Current question: {q}")
    
    # Compose full prompt
    full_prompt = "\n\n".join(prompt_parts)
    
    # Call existing LLM synthesis path unchanged
    answer = None
    print(os.getenv("OPENAI_API_KEY"))
    if req.use_llm :
        try:
            print("Load the model for LLM")
            llm = ChatOllama(model="myname_model:latest")
            response = llm.invoke(full_prompt)
            print(response.content)
            answer = response.content

        except Exception as e:
            print("something went wrong", e)
            answer = None
    
    # Persist messages to SQLiteMemory
    if sqlite_memory is not None and answer is not None:
        try:
            sqlite_memory.save_message(student_id, 'user', q)
            sqlite_memory.save_message(student_id, 'assistant', answer)
        except Exception as e:
            logging.warning(f"Failed to save messages to SQLiteMemory: {e}")
    
    # Add QA pair to SemanticMemory
    if semantic_memory is not None and answer is not None:
        try:
            semantic_memory.add_memory(student_id, 'user', q)
            semantic_memory.add_memory(student_id, 'assistant', answer)
        except Exception as e:
            logging.warning(f"Failed to add to SemanticMemory: {e}")
    
    # Return enhanced response
    response_data = {"query": q, "answer": answer, "retrieved": results}
    
    # Add optional fields only when memory is available
    if retrieved_qa:
        response_data["retrieved_qa"] = retrieved_qa
    if chat_history:
        response_data["chat_history"] = chat_history
    
    return response_data

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
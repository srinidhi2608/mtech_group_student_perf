from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional
import os
import logging

from src.data_processing import load_companies, extract_company_skills
from src.recommender import load_courses_catalog, recommend_courses_for_company
from src.pdf_ingest import ingest_pdf
from src.embeddings import get_embeddings_model
from src.vectorstore import get_vector_store
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Perf & Recommender API")

MODEL_PREPROCESSOR_PATH = "models/student_perf_preprocessor.joblib"
MODEL_LGBM_PATH = "models/student_perf_lgbm.joblib"
COMPANIES_CSV = "Glassdoor_Salary_Cleaned_Version.csv"
COURSES_CSV = "data/courses_catalog.csv"

try:
    preprocessor = joblib.load(MODEL_PREPROCESSOR_PATH)
    model = joblib.load(MODEL_LGBM_PATH)
except Exception:
    preprocessor = None
    model = None

try:
    companies_df = load_companies(COMPANIES_CSV)
    company_skill_map, _ = extract_company_skills(companies_df, text_col='Job Description', top_k=30)
except Exception:
    company_skill_map = {}
    companies_df = pd.DataFrame()

courses_df = load_courses_catalog(COURSES_CSV)

# Initialize embeddings and vector store
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    embeddings_model = get_embeddings_model()
    vector_store = get_vector_store(embedding_dim=embeddings_model.get_embedding_dim())
    
    # Add course catalog to vector store if not already present
    if vector_store.get_stats()["total_vectors"] == 0 and not courses_df.empty:
        logger.info("Indexing course catalog...")
        course_texts = []
        course_docs = []
        for _, row in courses_df.iterrows():
            text = f"{row['course_name']}: {row['skills']}"
            course_texts.append(text)
            course_docs.append({
                "text": text,
                "source": "course_catalog",
                "type": "course",
                "course_id": row['course_id'],
                "course_name": row['course_name'],
                "skills": row['skills'],
                "url": row.get('url', '')
            })
        embeddings = embeddings_model.embed_texts(course_texts)
        vector_store.add_documents(embeddings, course_docs)
        vector_store.save()
        logger.info(f"Indexed {len(course_texts)} courses")
except Exception as e:
    logger.error(f"Error initializing embeddings/vector store: {e}")
    embeddings_model = None
    vector_store = None

class StudentInput(BaseModel):
    student_id: Optional[int]
    name: Optional[str]
    branch: str
    gpa: float
    projects: Optional[str] = ""

class RecommendInput(BaseModel):
    target_company: Optional[str] = None
    completed_courses: Optional[List[str]] = []
    target_ctc: Optional[float] = None

class ChatInput(BaseModel):
    query: str
    top_k: int = 5

class SearchInput(BaseModel):
    query: str
    top_k: int = 10

@app.post("/predict")
def predict(student: StudentInput):
    if preprocessor is None or model is None:
        return {"error": "Model not trained yet. Run training first."}
    df = pd.DataFrame([student.dict()])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    return {"predicted_gpa": float(pred)}

@app.post("/recommend")
def recommend(data: RecommendInput):
    completed = data.completed_courses or []
    if data.target_company:
        recs = recommend_courses_for_company(data.target_company, company_skill_map, courses_df, completed, top_n=6)
        return {"company": data.target_company, "recommendations": recs}
    elif data.target_ctc:
        if companies_df.empty:
            return {"error": "Company database not available."}
        companies_df['avg_salary'] = (companies_df.get('min_salary',0).fillna(0).astype(float) + companies_df.get('max_salary',0).fillna(0).astype(float))/2
        idx = (companies_df['avg_salary'] - data.target_ctc).abs().idxmin()
        company_name = companies_df.loc[idx, 'Company Name']
        recs = recommend_courses_for_company(company_name, company_skill_map, courses_df, completed, top_n=6)
        return {"company": company_name, "recommendations": recs}
    else:
        return {"error": "Provide target_company or target_ctc."}

@app.get("/")
def root():
    return {"status": "ok", "note": "Use /predict and /recommend endpoints."}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and index a PDF file for RAG.
    """
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are supported"}
    
    if embeddings_model is None or vector_store is None:
        return {"error": "Embeddings/vector store not initialized"}
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Ingest and chunk PDF
        documents = ingest_pdf(file_path)
        
        if not documents:
            return {"error": "No content extracted from PDF"}
        
        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = embeddings_model.embed_texts(texts)
        
        # Add to vector store
        vector_store.add_documents(embeddings, documents)
        vector_store.save()
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(documents),
            "message": f"PDF {file.filename} indexed successfully"
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": str(e)}

@app.post("/chat")
def chat(data: ChatInput):
    """
    Chat endpoint with RAG support.
    Retrieves relevant document chunks and course entries, then returns:
    1. Retrieved snippets
    2. Synthesized answer (template-based if no OPENAI_API_KEY, otherwise LLM-based)
    """
    if embeddings_model is None or vector_store is None:
        return {"error": "Embeddings/vector store not initialized"}
    
    try:
        # Generate query embedding
        query_embedding = embeddings_model.embed_text(data.query)
        
        # Search vector store
        distances, results = vector_store.search(query_embedding, k=data.top_k)
        
        # Separate PDF chunks and course entries
        pdf_chunks = [r for r in results if r.get("type") != "course"]
        course_entries = [r for r in results if r.get("type") == "course"]
        
        # Build context from retrieved documents
        context_parts = []
        if pdf_chunks:
            context_parts.append("**From uploaded PDFs:**")
            for i, chunk in enumerate(pdf_chunks[:3], 1):
                context_parts.append(f"{i}. {chunk['text'][:200]}... (from {chunk['source']})")
        
        if course_entries:
            context_parts.append("\n**Relevant courses:**")
            for course in course_entries[:3]:
                context_parts.append(f"- {course['course_name']}: {course['skills']}")
        
        context = "\n".join(context_parts)
        
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            # Try to use OpenAI (optional, requires openai package)
            try:
                import openai
                openai.api_key = openai_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful student coach assistant. Use the provided context to answer questions."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {data.query}"}
                    ],
                    max_tokens=300
                )
                answer = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}, falling back to template")
                answer = _generate_template_answer(data.query, pdf_chunks, course_entries)
        else:
            # Generate template-based answer
            answer = _generate_template_answer(data.query, pdf_chunks, course_entries)
        
        return {
            "query": data.query,
            "answer": answer,
            "retrieved_chunks": len(results),
            "pdf_sources": [chunk['source'] for chunk in pdf_chunks],
            "recommended_courses": [course['course_name'] for course in course_entries],
            "context": context
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {"error": str(e)}

@app.post("/search")
def search(data: SearchInput):
    """
    Search for relevant documents and courses.
    """
    if embeddings_model is None or vector_store is None:
        return {"error": "Embeddings/vector store not initialized"}
    
    try:
        # Generate query embedding
        query_embedding = embeddings_model.embed_text(data.query)
        
        # Search vector store
        distances, results = vector_store.search(query_embedding, k=data.top_k)
        
        # Format results
        formatted_results = []
        for i, (dist, doc) in enumerate(zip(distances, results)):
            formatted_results.append({
                "rank": i + 1,
                "score": float(dist),
                "type": doc.get("type", "pdf"),
                "source": doc.get("source", "unknown"),
                "text": doc.get("text", "")[:300],
                "metadata": doc.get("metadata", {})
            })
        
        return {
            "query": data.query,
            "results": formatted_results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return {"error": str(e)}

def _generate_template_answer(query: str, pdf_chunks: List[Dict], course_entries: List[Dict]) -> str:
    """
    Generate a template-based answer when no LLM is available.
    """
    answer_parts = []
    
    if pdf_chunks:
        answer_parts.append(f"Based on the uploaded documents, I found {len(pdf_chunks)} relevant sections.")
        answer_parts.append(f"The most relevant information comes from: {', '.join(set(c['source'] for c in pdf_chunks[:3]))}.")
    
    if course_entries:
        answer_parts.append(f"\nI recommend checking out these courses:")
        for course in course_entries[:3]:
            answer_parts.append(f"- {course['course_name']} (covers: {course['skills']})")
    
    if not answer_parts:
        answer_parts.append("I couldn't find specific information about your query in the indexed documents.")
        answer_parts.append("Try uploading relevant PDFs or asking about available courses.")
    
    return " ".join(answer_parts)

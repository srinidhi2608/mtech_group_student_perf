# Student Performance & Course Recommender — Enhanced with RAG

This repository branch contains an advanced prototype pipeline that:

- Predicts student performance (GPA) using a LightGBM model with Optuna tuning.
- Extracts skills from company job descriptions (Glassdoor dataset) and builds company skill profiles.
- Recommends courses (from a course catalog) to students so they can close skill gaps for a target company or target CTC.
- **NEW: PDF ingestion pipeline** - Upload and index educational PDFs for semantic search.
- **NEW: RAG-powered chat** - Chat with your documents using retrieval-augmented generation.
- **NEW: Vector search** - FAISS-based semantic search over documents and courses.
- Exposes a FastAPI backend with /predict, /recommend, /upload_pdf, /chat, and /search endpoints.
- Provides an enhanced Streamlit frontend with tabs for predictions, chat, and PDF uploads.

This README explains how to run the system locally (development), train the model, and use the UI/API including the new PDF/chat features.

## Contents
- src/: data processing, training, recommender logic, **PDF ingestion, embeddings, vector store**
- app/: FastAPI backend with **RAG endpoints**
- frontend/: Enhanced Streamlit UI with **chat and PDF upload**
- data/: sample courses catalog
- models/: training artifacts, **FAISS index and metadata** (created after training/indexing)
- uploads/: uploaded PDF files (created at runtime)
- requirements.txt: includes **sentence-transformers, faiss-cpu, PyMuPDF, python-multipart**

## Prerequisites
- Python 3.8+ (3.10 recommended)
- Git
- ~8 GB free disk space; more during training/embedding installs
- (Optional) Docker if you prefer containerized runs

## 1) Clone & checkout the branch

```bash
git clone https://github.com/srinidhi2608/mtech_group_student_perf.git
cd mtech_group_student_perf
git checkout feature/advanced-ml-recommender
```

## 2) Create and activate a virtual environment

```bash
py -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1
```

## 3) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The requirements include NLP & transformer libraries; installing sentence-transformers may pull a few hundred MB.
- **New dependencies**: faiss-cpu (vector search), PyMuPDF (PDF parsing), python-multipart (file uploads).
- If you face issues during installation, try upgrading pip and wheel, or install troublesome packages separately.

## 4) Download NLTK resources (one-time)

```bash
py -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 5) Prepare data

- Student dataset:
  Place your student CSV at the repo root and name it `students_sample.csv` (or update `src/train.py` to point to your file).
  Expected minimal columns (example):
  ```csv
  student_id,name,branch,gpa,projects
  ```
- Company dataset:
  Ensure `Glassdoor_Salary_Cleaned_Version.csv` is present at repo root (the recommender extracts skills from its `Job Description` column).
- Courses catalog:
  There is a sample at `data/courses_catalog.csv`. Expand this with course descriptions and skills (comma-separated).

## 6) Train the student performance model (creates models/)

```bash
py -m src.train
```

What this does:
- Loads `students_sample.csv`
- Builds a TF-IDF + one-hot preprocessing pipeline for `projects` + `branch`
- Runs an Optuna search (default 20 trials) to tune LightGBM hyperparameters
- Saves artifacts to `models/`:
  - `models/student_perf_preprocessor.joblib`
  - `models/student_perf_lgbm.joblib`

Tips:
- For a real run use your full dataset and increase `n_trials` in `src/train.py`
- If your target is not literal GPA, adjust `prepare_target()` in `src/train.py`

## 7) Start the backend API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- GET  /                         -> status message
- POST /predict                  -> body: {"branch":"...", "gpa":..., "projects":"..."}
                                   returns: {"predicted_gpa": ...}
- POST /recommend                -> body examples:
                                   {"target_company":"Wish","completed_courses":["Intro to Python"]}
                                   {"target_ctc":120000,"completed_courses":[]}
                                   returns: {"company": "<name>", "recommendations":[{course info}]}
- **POST /upload_pdf**           -> Upload a PDF file for indexing
                                   returns: {"status": "success", "chunks_indexed": N}
- **POST /chat**                 -> body: {"query": "your question", "top_k": 5}
                                   returns: RAG-powered answer with retrieved context
- **POST /search**               -> body: {"query": "search term", "top_k": 10}
                                   returns: semantic search results from indexed documents

**New RAG Features:**
- The `/chat` endpoint uses retrieval-augmented generation (RAG)
- It retrieves top-k document chunks and course entries via FAISS
- If `OPENAI_API_KEY` environment variable is set, it uses GPT-3.5 for answers
- Otherwise, it returns a synthesized template-based answer
- Uploaded PDFs are saved to `uploads/` and indexed automatically
- FAISS index and metadata are persisted to `models/faiss.index` and `models/faiss_meta.json`
- The system is defensive: if no index exists at startup, it initializes blank structures

## 8) Run the Streamlit frontend (separate terminal)

```bash
streamlit run frontend/app.py
```

**Enhanced UI with three tabs:**

1. **Predict & Recommend** (original features):
   - Use the sidebar to enter student info, target company or CTC, completed courses
   - Click "Ask Coach" to get a predicted GPA and recommended courses

2. **Chat with PDFs** (new):
   - Interactive chat interface to ask questions about uploaded documents
   - Get answers with context from PDFs and course recommendations
   - View sources and recommended courses

3. **Upload PDFs** (new):
   - Upload educational PDFs, textbooks, or course materials
   - System automatically indexes the content for semantic search
   - Search functionality to find specific topics across all documents

## 9) Example API usage (curl)

Predict:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"branch":"Computer Science","gpa":8.2,"projects":"Image classifier"}'
```

Recommend:
```bash
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d '{"target_company":"Wish","completed_courses":["Intro to Python"]}'
```

**Upload PDF:**
```bash
curl -X POST "http://localhost:8000/upload_pdf" -F "file=@/path/to/document.pdf"
```

**Chat:**
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query":"What are the best courses for machine learning?","top_k":5}'
```

**Search:**
```bash
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query":"neural networks","top_k":10}'
```

## 10) Test the new PDF/RAG features

Run the sanity tests to verify the components work:
```bash
python test_sanity.py
```

This will test:
- Text chunking functionality
- Embeddings model (sentence-transformers)
- Vector store (FAISS) with save/load

All tests should pass before using the system.

## 11) Using OpenAI for better chat responses (optional)

To enable GPT-powered answers in the chat endpoint:
```bash
export OPENAI_API_KEY="your-api-key-here"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If no API key is provided, the system will use template-based responses that still work well for basic queries.

## 12) Docker (optional)

- I can add a Dockerfile on request. Typical steps:
  - Build image with Python dependencies and copy repo
  - Create separate services for backend and frontend (or a single service if simple)
  - Use docker-compose for multi-service local setup (backend + frontend)

## 13) Next improvements / recommended work

- Improve data features:
  - Add term-wise grades, attendance, credits, internships, placements, domain projects.
  - Encode time series (RNN/Transformer) if you have multi-term trajectories.
- Better NLP matching:
  - **✓ DONE**: Use sentence-transformers to embed job descriptions and course descriptions and compute semantic similarity.
  - Use named-entity recognition or a curated job-skill mapping for higher precision.
- **RAG enhancements**:
  - Add conversation history for multi-turn chat
  - Implement hybrid search (keyword + semantic)
  - Add re-ranking for better result quality
  - Support more document formats (DOCX, HTML, etc.)
- Model improvements:
  - Try neural networks (Keras/TensorFlow) for richer feature interactions, or ensemble models.
  - Calibrate and validate model with cross-validation and hold-out sets.
- Productionize:
  - Add authentication, rate limiting, logging, metrics.
  - Add CI, unit tests, and E2E tests.
  - Containerize and deploy to cloud (recommended: AWS/GCP/Azure, Render, or Heroku for simple deployments).
- UX improvements:
  - A richer React-based chat UI with conversation context and user accounts.
  - Show learning paths, course durations, estimated time to reach target skills.
  - **✓ DONE**: Multi-tab Streamlit interface with chat and PDF upload.

## 14) Troubleshooting

- FastAPI error about model artifacts: ensure `models/` has the .joblib files after training.
- Heavy install steps: install `transformers`/`sentence-transformers` only if you plan to use embeddings.
- **PDF upload fails**: Check that `uploads/` directory exists and is writable.
- **Chat returns errors**: Ensure embeddings model loaded successfully (check logs).
- **Vector store empty**: Upload at least one PDF or ensure course catalog is indexed.
- If the recommender returns no matches: expand `data/courses_catalog.csv` and ensure skills are mapped.

## 15) License & credits

- This is prototype code. Before production use, review, test, and adapt licensing as needed.

---

## 16) New Feature: Semester 6 Prediction and Company Matcher

### Training the Sem_6 Prediction Model

The new training script in `src/train.py` trains a LightGBM model to predict Semester 6 scores based on:
- Previous semester scores (Sem_1 through Sem_5)
- PUC scores
- Attendance percentage
- Study hours per day
- Sleep hours per day
- Courses completed (multi-hot encoded)

To train the model:
```bash
python -m src.train
```

This creates `models/sem6_pred.joblib` containing the trained model, MultiLabelBinarizer for courses, and feature column names.

### Using the /student_predict Endpoint

The new `/student_predict` endpoint provides:
1. **Sem_6 Score Prediction**: Based on student's academic and behavioral data
2. **Company Matching**: Recommends top 5 companies based on courses completed and their job requirements

**Request format:**
```bash
curl -X POST "http://localhost:8000/student_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "PUC_Score": 85.5,
    "Sem_1": 8.5,
    "Sem_2": 8.7,
    "Sem_3": 8.9,
    "Sem_4": 9.0,
    "Sem_5": 9.2,
    "Attendance_%": 88.0,
    "Study_Hours_Per_Day": 6.0,
    "Sleep_Hours_Per_Day": 7.0,
    "Courses_Completed": "Python,Java,Machine Learning"
  }'
```

**Response format:**
```json
{
  "predicted_Sem_6": 8.95,
  "company_matches": [
    {
      "company": "Google",
      "overlap": 12,
      "avg_salary": 150000
    },
    {
      "company": "Microsoft",
      "overlap": 10,
      "avg_salary": 145000
    }
  ]
}
```

The company matcher uses TF-IDF vectorization on job descriptions to extract skill keywords and matches them against the student's completed courses. Results are ranked by skill overlap and average salary.

### Quick Start

1. Ensure you have the required CSV files:
   - `simple_student_dataset.csv` (for training)
   - `Glassdoor_Salary_Cleaned_Version.csv` (for company matching)

2. Train the model:
   ```bash
   python -m src.train
   ```

3. Start the API server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Test the endpoint:
   ```bash
   curl -X POST "http://localhost:8000/student_predict" \
     -H "Content-Type: application/json" \
     -d '{"PUC_Score":75,"Sem_1":8,"Sem_2":8,"Sem_3":8,"Sem_4":8,"Sem_5":8,"Attendance_%":85,"Study_Hours_Per_Day":5,"Sleep_Hours_Per_Day":7,"Courses_Completed":"Python"}'
   ```

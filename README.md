# Student Performance & Course Recommender — Prototype

This repository branch contains a prototype pipeline to:

- Predict student performance (GPA) using a LightGBM model with Optuna tuning.
- Extract skills from company job descriptions (Glassdoor dataset) and build company skill profiles.
- Recommend courses (from a course catalog) to students so they can close skill gaps for a target company or target CTC.
- Expose a FastAPI backend with /predict and /recommend endpoints.
- Provide a lightweight Streamlit frontend (chat-like UI) for students.

This README explains how to run the system locally (development), train the model, and use the UI/API.

## Contents
- src/: data processing, training, recommender logic
- app/: FastAPI backend
- frontend/: Streamlit UI
- data/: sample courses catalog
- models/: training artifacts (created after training)
- requirements.txt

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
python -m venv .venv
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
- If you face issues during installation, try upgrading pip and wheel, or install troublesome packages separately.

## 4) Download NLTK resources (one-time)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
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
python -m src.train
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

## 8) Run the Streamlit frontend (separate terminal)

```bash
streamlit run frontend/app.py
```

- Use the sidebar to enter student info, target company or CTC, completed courses.
- Click "Ask Coach" to get a predicted GPA and recommended courses.

## 9) Example API usage (curl)

Predict:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"branch":"Computer Science","gpa":8.2,"projects":"Image classifier"}'
```

Recommend:
```bash
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d '{"target_company":"Wish","completed_courses":["Intro to Python"]}'
```

## 10) Docker (optional)

- I can add a Dockerfile on request. Typical steps:
  - Build image with Python dependencies and copy repo
  - Create separate services for backend and frontend (or a single service if simple)
  - Use docker-compose for multi-service local setup (backend + frontend)

## 11) Next improvements / recommended work

- Improve data features:
  - Add term-wise grades, attendance, credits, internships, placements, domain projects.
  - Encode time series (RNN/Transformer) if you have multi-term trajectories.
- Better NLP matching:
  - Use sentence-transformers to embed job descriptions and course descriptions and compute semantic similarity.
  - Use named-entity recognition or a curated job-skill mapping for higher precision.
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

## 12) Troubleshooting

- FastAPI error about model artifacts: ensure `models/` has the .joblib files after training.
- Heavy install steps: install `transformers`/`sentence-transformers` only if you plan to use embeddings.
- If the recommender returns no matches: expand `data/courses_catalog.csv` and ensure skills are mapped.

## 13) License & credits

- This is prototype code. Before production use, review, test, and adapt licensing as needed.

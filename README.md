# Student Performance & Course Recommender (Prototype)

This branch adds a prototype to predict student performance and recommend courses to improve employability for a target company or target CTC. It includes:

- A LightGBM model with Optuna hyperparameter tuning (src/train.py).
- A TF-IDF-based skill extractor built from Glassdoor job descriptions to profile companies (src/data_processing.py).
- A content-based course recommender that maps company skills to a course catalog (src/recommender.py).
- A FastAPI backend exposing /predict and /recommend endpoints (app/main.py).
- A Streamlit frontend with a chat-like UI for students (frontend/app.py).

## Setup (local)

1. Clone repository and checkout the branch (already created):

   git clone https://github.com/srinidhi2608/mtech_group_student_perf.git
   cd mtech_group_student_perf
   git checkout feature/advanced-ml-recommender

2. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install --upgrade pip
   pip install -r requirements.txt

3. Download NLTK data (required once):

   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

4. Prepare datasets:

   - Place your student dataset CSV at the repo root and name it `students_sample.csv` (or update src/train.py to point to your file).
   - Make sure the Glassdoor dataset `Glassdoor_Salary_Cleaned_Version.csv` is in the repo root. The recommender extracts skills from the `Job Description` column.

5. Train the model (this creates `models/` artifacts):

   python -m src.train

   This will run Optuna for a small number of trials and write:
   - models/student_perf_preprocessor.joblib
   - models/student_perf_lgbm.joblib

6. Run the backend API:

   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

7. Run the Streamlit frontend (in a separate terminal):

   streamlit run frontend/app.py

8. Use the UI:

   - Fill the student info in the sidebar and click "Ask Coach".
   - The UI will call the backend to predict next GPA and recommend courses.

## Notes & next steps

- Improve the student model with more features (attendance, term-wise grades, extra-curriculars).
- Replace TF-IDF matching by semantic embeddings (sentence-transformers) for better skill matching.
- Add authentication to the API and rate-limiting for public deployments.
- Add unit tests, CI, and a Dockerfile for reproducible deployments.

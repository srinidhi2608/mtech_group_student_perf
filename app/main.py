from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional

from src.data_processing import load_companies, extract_company_skills
from src.recommender import load_courses_catalog, recommend_courses_for_company

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

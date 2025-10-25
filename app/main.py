from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional
from src.company_matcher import load_companies, build_company_skill_profiles, match_student_courses_to_companies
import os

app = FastAPI()

MODEL_PATH = "models/sem6_pred.joblib"
sem6_model = None
sem6_mlb = None
sem6_feature_cols = None
if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        sem6_model = artifact.get('model')
        sem6_mlb = artifact.get('mlb')
        sem6_feature_cols = artifact.get('feature_cols')
    except Exception:
        sem6_model = None
        sem6_mlb = None
        sem6_feature_cols = None

# Load companies and build TF-IDF profiles (safe, non-fatal on errors)
companies_df = pd.DataFrame()
company_profiles = {}
try:
    companies_df = load_companies("Glassdoor_Salary_Cleaned_Version.csv")
    company_profiles, _ = build_company_skill_profiles(companies_df, text_col='Job Description')
except Exception:
    # keep empty structures; matching will return empty list
    companies_df = pd.DataFrame()
    company_profiles = {}

class StudentPayload(BaseModel):
    PUC_Score: float
    Sem_1: float
    Sem_2: float
    Sem_3: float
    Sem_4: float
    Sem_5: float
    Attendance_%: float
    Study_Hours_Per_Day: float
    Sleep_Hours_Per_Day: float
    Courses_Completed: Optional[str] = ""

@app.post('/student_predict')
def student_predict(payload: StudentPayload):
    if sem6_model is None or sem6_mlb is None or sem6_feature_cols is None:
        raise HTTPException(status_code=500, detail="Model artifact not available. Train model first and place models/sem6_pred.joblib")
    data = payload.dict()
    courses = [c.strip().lower() for c in data.get('Courses_Completed','').split(',') if c.strip()]
    try:
        courses_vec = sem6_mlb.transform([courses])
    except Exception:
        # If transform fails (mlb mismatch), produce zero vector of appropriate length
        import numpy as np
        courses_vec = np.zeros((1, len(sem6_mlb.classes_)))
    courses_df = pd.DataFrame(courses_vec, columns=[f"course_{c}" for c in sem6_mlb.classes_])
    feature_vals = pd.DataFrame([{k: data[k] for k in ['PUC_Score','Sem_1','Sem_2','Sem_3','Sem_4','Sem_5','Attendance_%','Study_Hours_Per_Day','Sleep_Hours_Per_Day']}])
    X = pd.concat([feature_vals.reset_index(drop=True), courses_df.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=sem6_feature_cols).fillna(0)
    pred = float(sem6_model.predict(X)[0])
    matches = []
    try:
        matches = match_student_courses_to_companies(courses, company_profiles, companies_df, top_n=5)
    except Exception:
        matches = []
    return {"predicted_Sem_6": pred, "company_matches": matches}

@app.get('/')
def root():
    return {"status": "ok"}

import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_courses_catalog(path="data/courses_catalog.csv"):
    """
    Expect CSV with columns: course_id, title, description, skills (comma-separated)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'skills' not in df.columns:
        # create skills from title/description using naive split if missing
        df['skills'] = df.get('description','').fillna('').apply(lambda x: ','.join([]))
    df['skills_list'] = df['skills'].fillna('').apply(lambda s: [x.strip().lower() for x in s.split(',') if x.strip()])
    return df

def recommend_courses_for_skills(target_skills: List[str], courses_df: pd.DataFrame, top_n=5):
    """
    Simple matching: count overlap between course skills and target_skills.
    """
    tset = set([s.lower() for s in target_skills])
    candidates = []
    for _, row in courses_df.iterrows():
        cskills = set(row.get('skills_list', []))
        overlap = len(tset & cskills)
        candidates.append((overlap, row))
    candidates_sorted = sorted(candidates, key=lambda x: (x[0],), reverse=True)
    recs = []
    for score, row in candidates_sorted[:top_n]:
        recs.append({
            "course_id": row.get('course_id'),
            "title": row.get('title'),
            "description": row.get('description'),
            "skills": row.get('skills'),
            "match_score": int(score)
        })
    return recs
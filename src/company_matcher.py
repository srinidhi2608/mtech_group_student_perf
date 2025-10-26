import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple

def load_companies(path="Glassdoor_Salary_Cleaned_Version.csv"):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # try to compute avg_salary if min/max exist
    if 'min_salary' in df.columns and 'max_salary' in df.columns:
        df['min_salary'] = pd.to_numeric(df['min_salary'], errors='coerce')
        df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')
        df['avg_salary'] = df[['min_salary','max_salary']].mean(axis=1)
    else:
        df['avg_salary'] = np.nan
    return df

def build_skill_profile_from_flags(companies_df, skill_columns: List[str]):
    """
    Build company skill profile using explicit boolean columns like python_yn, R_yn, spark, aws, excel.
    Returns mapping company -> set(skills)
    """
    profiles = {}
    for _, row in companies_df.iterrows():
        skills = set()
        for col in skill_columns:
            if col in companies_df.columns:
                val = row.get(col, 0)
                try:
                    if str(val).strip().lower() in ('1','true','yes','y'):
                        skills.add(col.replace('_yn','').lower())
                except Exception:
                    pass
        profiles[row.get('Company Name', f"company_{_}")] = skills
    return profiles

def build_company_skill_profiles(companies_df, text_col='Job Description', max_features=2000):
    """
    Fallback: build TF-IDF keyword profiles from Job Description
    """
    docs = companies_df[text_col].fillna('').astype(str).values
    vect = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(docs)
    feature_names = vect.get_feature_names_out()
    profiles = {}
    for idx, comp in companies_df.iterrows():
        row = X[idx].toarray().ravel()
        top_idx = np.argsort(row)[::-1][:40]
        keywords = [feature_names[i] for i in top_idx if row[i] > 0]
        profiles[comp['Company Name']] = set([k.lower() for k in keywords])
    return profiles, vect

def match_student_to_companies_by_flags(student_skills: List[str], companies_df: pd.DataFrame, skill_columns: List[str], top_n=5):
    """
    If Glassdoor dataset contains explicit skill flags, use that to match.
    """
    profiles = build_skill_profile_from_flags(companies_df, skill_columns)
    sset = set([s.lower() for s in student_skills])
    matches = []
    for comp, skills in profiles.items():
        overlap = len(sset & set(skills))
        row = companies_df[companies_df['Company Name'] == comp]
        avg_salary = float(row['avg_salary'].iloc[0]) if (not row.empty and 'avg_salary' in row.columns and not pd.isna(row['avg_salary'].iloc[0])) else None
        matches.append({'company': comp, 'overlap': int(overlap), 'avg_salary': avg_salary})
    matches_sorted = sorted(matches, key=lambda x: (x['overlap'], x['avg_salary'] if x['avg_salary'] is not None else 0), reverse=True)
    return matches_sorted[:top_n]

def match_student_to_companies_by_text(student_skills: List[str], company_profiles: Dict[str,set], companies_df: pd.DataFrame, top_n=5):
    sset = set([s.lower() for s in student_skills])
    matches = []
    for comp, kws in company_profiles.items():
        overlap = len(sset & set(kws))
        row = companies_df[companies_df['Company Name'] == comp]
        avg_salary = float(row['avg_salary'].iloc[0]) if (not row.empty and 'avg_salary' in row.columns and not pd.isna(row['avg_salary'].iloc[0])) else None
        matches.append({'company': comp, 'overlap': int(overlap), 'avg_salary': avg_salary})
    matches_sorted = sorted(matches, key=lambda x: (x['overlap'], x['avg_salary'] if x['avg_salary'] is not None else 0), reverse=True)
    return matches_sorted[:top_n]
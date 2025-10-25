import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

def load_companies(path="Glassdoor_Salary_Cleaned_Version.csv"):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if 'min_salary' in df.columns and 'max_salary' in df.columns:
        df['avg_salary'] = (pd.to_numeric(df['min_salary'], errors='coerce').fillna(0) + pd.to_numeric(df['max_salary'], errors='coerce').fillna(0)) / 2
    else:
        df['avg_salary'] = np.nan
    return df

def build_company_skill_profiles(companies_df, text_col='Job Description', max_features=2000):
    docs = companies_df[text_col].fillna('').astype(str).values
    vect = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(docs)
    feature_names = vect.get_feature_names_out()
    profiles = {}
    for idx, comp in companies_df.iterrows():
        row = X[idx].toarray().ravel()
        top_idx = np.argsort(row)[::-1][:50]
        keywords = [feature_names[i] for i in top_idx if row[i] > 0]
        profiles[comp['Company Name']] = keywords
    return profiles, vect

def match_student_courses_to_companies(student_courses: List[str], company_profiles: Dict[str, List[str]], companies_df: pd.DataFrame, top_n=5):
    student_skills = set([s.strip().lower() for s in student_courses])
    matches = []
    for company, keywords in company_profiles.items():
        kws = set([k.lower() for k in keywords])
        overlap = len(student_skills & kws)
        avg_salary = None
        row = companies_df[companies_df['Company Name'] == company]
        if not row.empty and 'avg_salary' in row.columns:
            avg_salary = float(row.iloc[0].get('avg_salary', np.nan)) if not pd.isna(row.iloc[0].get('avg_salary', np.nan)) else None
        matches.append({'company': company, 'overlap': int(overlap), 'avg_salary': avg_salary})
    matches_sorted = sorted(matches, key=lambda x: (x['overlap'], x['avg_salary'] if x['avg_salary'] is not None else 0), reverse=True)
    return matches_sorted[:top_n]

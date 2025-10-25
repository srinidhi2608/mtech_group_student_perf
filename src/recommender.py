import pandas as pd
from typing import List
import numpy as np

def load_courses_catalog(path="data/courses_catalog.csv"):
    return pd.read_csv(path)

def map_courses_by_skill(courses_df):
    skill_map = {}
    for _, row in courses_df.iterrows():
        skills = [s.strip().lower() for s in str(row['skills']).split(',') if s.strip()]
        for sk in skills:
            skill_map.setdefault(sk, []).append({
                'course_id': row['course_id'],
                'course_name': row['course_name'],
                'skills': skills,
                'url': row.get('url', '')
            })
    return skill_map

def recommend_courses_for_company(company_name, company_skill_map:dict, courses_df, completed_courses:List[str], top_n=5):
    required_skills = company_skill_map.get(company_name, [])
    required_skills = [s.lower() for s in required_skills]
    courses = courses_df.copy()
    def score_course(skills_text):
        skills = [s.strip().lower() for s in str(skills_text).split(',') if s.strip()]
        return len(set(skills) & set(required_skills))
    courses['score'] = courses['skills'].apply(score_course)
    courses = courses[~courses['course_name'].isin(completed_courses)]
    recommended = courses.sort_values(['score'], ascending=False)
    top = recommended[recommended['score']>0].head(top_n)
    if top.shape[0]==0:
        top = courses.head(top_n)
    return top[['course_id','course_name','skills','url','score']].to_dict(orient='records')

def completed_skills_from_courses(completed_courses, courses_df):
    skills = set()
    for c in completed_courses:
        row = courses_df[courses_df['course_name']==c]
        if not row.empty:
            s = row.iloc[0]['skills']
            for sk in str(s).split(','):
                skills.add(sk.strip().lower())
    return skills

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from nltk.corpus import stopwords

# If running for first time you may need:
# import nltk
# nltk.download('punkt'); nltk.download('stopwords')

STOP = set(stopwords.words('english')) if True else set()

def load_students(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_companies(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def simple_text_clean(text):
    if pd.isna(text):
        return ""
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def extract_company_skills(companies_df, text_col='Job Description', top_k=40):
    """
    Build a mapping company -> list of top skill keywords extracted using TF-IDF
    Returns: skills_per_company (dict) and fitted vectorizer
    """
    df = companies_df.copy()
    df['doc'] = df[text_col].fillna('').apply(simple_text_clean)
    agg = df.groupby('Company Name')['doc'].apply(lambda docs: ' '.join(docs)).reset_index()
    docs = agg['doc'].values
    vect = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(docs)
    feature_names = np.array(vect.get_feature_names_out())
    skills_per_company = {}
    for i, company in enumerate(agg['Company Name'].values):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_k]
        skills = [feature_names[idx] for idx in top_idx if feature_names[idx] not in STOP]
        skills_per_company[company] = skills
    return skills_per_company, vect

def build_student_feature_pipeline(student_df):
    """
    Build and fit a simple preprocessing pipeline for student features.
    Assumes student_df has: student_id, name, branch, gpa, projects
    """
    df = student_df.copy()
    df['projects'] = df['projects'].fillna('')
    df['branch'] = df['branch'].fillna('Unknown')
    project_vect = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
    onehot_branch = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer(transformers=[
        ('projects', project_vect, 'projects'),
        ('branch', onehot_branch, ['branch'])
    ], remainder='drop')
    preprocessor.fit(df)
    return preprocessor

def featurize_students(preprocessor, df):
    df2 = df.copy()
    df2['projects'] = df2['projects'].fillna('')
    df2['branch'] = df2['branch'].fillna('Unknown')
    X = preprocessor.transform(df2)
    return X

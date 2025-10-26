import requests
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from typing import List

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Student - Chatbot", layout="wide")
st.title("🎓 Student Page — Predict, Recommend & Chat")


# Attempt to import local helpers if available (used for company matching fallback)
try:
    from src.company_matcher import load_companies, match_student_to_companies_by_flags, match_student_to_companies_by_text
except Exception:
    load_companies = None
    match_student_to_companies_by_flags = None
    match_student_to_companies_by_text = None


# Config
STUDENT_CSV = "simple_student_dataset.csv"
COMPANIES_CSV = "Glassdoor_Salary_Cleaned_Version.csv"
MODEL_ARTIFACT = "models/sem6_pred.joblib"
DEFAULT_API_URL = "http://localhost:8000/student_predict"

st.set_page_config(page_title="Student — Predict & Recommend", layout="wide")


def load_students(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Student dataset not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_companies_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # try to coerce salary columns
    for c in ("min_salary", "max_salary", "avg_salary"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_local_model(artifact_path: str):
    if not os.path.exists(artifact_path):
        return None
    try:
        artifact = joblib.load(artifact_path)
        return artifact
    except Exception as e:
        st.warning(f"Failed to load local model artifact: {e}")
        return None


def build_feature_row_from_payload(payload: dict, mlb, feature_cols: List[str]) -> pd.DataFrame:
    """
    Build a single-row dataframe with columns matching feature_cols using the payload and the mlb.
    payload keys expected: PUC_Score, Sem_1..Sem_5, Attendance_%, Study_Hours_Per_Day, Sleep_Hours_Per_Day, Courses_Completed
    """
    # numeric features
    base_feats = ['PUC_Score','Sem_1','Sem_2','Sem_3','Sem_4','Sem_5',
                  'Attendance_%','Study_Hours_Per_Day','Sleep_Hours_Per_Day']
    base_row = {k: float(payload.get(k, 0)) for k in base_feats}
    # courses
    courses = [c.strip().lower() for c in str(payload.get("Courses_Completed","")).split(',') if c.strip()]
    try:
        courses_vec = mlb.transform([courses])
        courses_cols = [f"course_{c}" for c in mlb.classes_]
        courses_df = pd.DataFrame(courses_vec, columns=courses_cols)
    except Exception:
        # mlb mismatch -> zeros
        courses_cols = [f"course_{c}" for c in (mlb.classes_ if mlb is not None else [])]
        courses_df = pd.DataFrame(np.zeros((1, len(courses_cols))), columns=courses_cols)

    X = pd.concat([pd.DataFrame([base_row]), courses_df.reset_index(drop=True)], axis=1)
    # reindex to feature_cols
    X = X.reindex(columns=feature_cols).fillna(0)
    return X


def predict_local_sem6(artifact, payload: dict):
    model = artifact.get("model")
    mlb = artifact.get("mlb")
    feature_cols = artifact.get("feature_cols")
    X = build_feature_row_from_payload(payload, mlb, feature_cols)
    try:
        pred = float(model.predict(X)[0])
        return pred
    except Exception as e:
        st.error(f"Local prediction failed: {e}")
        return None


def call_api_predict(api_url: str, payload: dict):
    try:
        r = requests.post(api_url, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json().get("predicted_Sem_6"), r.json()
        else:
            st.error(f"API returned {r.status_code}: {r.text}")
            return None, None
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None, None


def match_companies_for_student1(student_skills: List[str], companies_df: pd.DataFrame, top_n=5):
    # Prefer the flags-based matcher if available and columns present
    skill_flag_cols = ['python_yn','R_yn','spark','aws','excel']
    if not companies_df.empty and any(c in companies_df.columns for c in skill_flag_cols) and match_student_to_companies_by_flags is not None:
        try:
            return match_student_to_companies_by_flags(student_skills, companies_df, skill_flag_cols, top_n=top_n)
        except Exception:
            pass
    # fallback to text-based matcher if available
    if not companies_df.empty and match_student_to_companies_by_text is not None:
        # build text profiles from job description on the fly (the function in repo returns profiles when called elsewhere)
        from src.company_matcher import build_company_skill_profiles
        profiles, _ = build_company_skill_profiles(companies_df, text_col='Job Description')
        return match_student_to_companies_by_text(student_skills, profiles, companies_df, top_n=top_n)
    # Final fallback: empty
    return []

def match_companies_for_student(student_skills: List[str], companies_df: pd.DataFrame, top_n=3):
    """
    Improved matching:
    - If the Glassdoor dataset contains explicit skill flag columns (e.g., python_yn, r_yn, spark, aws, excel),
      compute a normalized flag-match score (proportion of student's skills present in the company's flags).
    - Otherwise, fall back to TF-IDF + cosine similarity between the student's skill terms and each company's Job Description.
    - Return top_n companies sorted by a combined score (skill_score weighted + salary normalization tie-breaker).
    Returned list: [{'company': <name>, 'score': <0-1>, 'overlap': <count>, 'avg_salary': <float_or_None>}...]
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # quick guard
    if companies_df is None or companies_df.empty:
        return []

    # normalize student skills
    s_skills = [s.strip().lower() for s in student_skills if s and s.strip()]
    if len(s_skills) == 0:
        # no student skills: return top companies by avg_salary (if available)
        if 'avg_salary' in companies_df.columns:
            df = companies_df.copy()
            df['avg_salary'] = pd.to_numeric(df['avg_salary'], errors='coerce').fillna(0)
            df = df.sort_values('avg_salary', ascending=False).head(top_n)
            return [{'company': row['Company Name'], 'score': 0.0, 'overlap': 0, 'avg_salary': float(row.get('avg_salary', np.nan))} for _, row in df.iterrows()]
        return []

    skill_flag_cols = [c for c in ['python_yn', 'r_yn', 'spark', 'aws', 'excel'] if c in companies_df.columns]
    matches = []

    # Prepare salary normalization (0..1) for tie-breaking
    if 'avg_salary' in companies_df.columns:
        salaries = pd.to_numeric(companies_df['avg_salary'], errors='coerce')
        min_sal = float(salaries.min(skipna=True)) if not salaries.isna().all() else 0.0
        max_sal = float(salaries.max(skipna=True)) if not salaries.isna().all() else 0.0
        def norm_salary(v):
            try:
                v = float(v)
                if max_sal > min_sal:
                    return (v - min_sal) / (max_sal - min_sal)
                return 0.0
            except Exception:
                return 0.0
    else:
        norm_salary = lambda v: 0.0

    if skill_flag_cols:
        # Map student skill tokens to candidate flag column names (simple heuristic)
        # e.g., 'python' -> 'python_yn', 'r' -> 'r_yn', 'spark' -> 'spark'
        flag_map = {}
        lower_cols = [c.lower() for c in companies_df.columns]
        for sk in s_skills:
            # exact matches first
            found = None
            for col in companies_df.columns:
                if sk in col.lower():  # match substring
                    found = col
                    break
            if found:
                flag_map[sk] = found
            else:
                # fallback mapping heuristics
                if sk == 'r':
                    # common flag might be 'R_yn' or 'r_yn'
                    for c in companies_df.columns:
                        if c.lower().startswith('r_'):
                            flag_map[sk] = c
                            break
                elif sk == 'python':
                    for c in companies_df.columns:
                        if 'python' in c.lower():
                            flag_map[sk] = c
                            break
                else:
                    # look for any column that contains the skill token
                    for c in companies_df.columns:
                        if sk in c.lower():
                            flag_map[sk] = c
                            break
        # Evaluate each company
        for _, row in companies_df.iterrows():
            match_count = 0
            for sk, col in flag_map.items():
                if col in row.index:
                    val = row.get(col)
                    try:
                        if str(val).strip().lower() in ('1', 'true', 'yes', 'y') or float(val) == 1.0:
                            match_count += 1
                    except Exception:
                        # non-numeric – treat non-empty/True-like strings
                        if str(val).strip().lower() in ('true','yes','y'):
                            match_count += 1
            skill_score = match_count / max(len(s_skills), 1)
            salary_score = norm_salary(row.get('avg_salary', np.nan))
            combined_score = 0.85 * skill_score + 0.15 * salary_score
            matches.append({'company': row.get('Company Name', f"company_{_}"),
                            'score': float(combined_score),
                            'overlap': int(match_count),
                            'avg_salary': float(row.get('avg_salary')) if pd.notna(row.get('avg_salary')) else None})
    else:
        # Text-based TF-IDF similarity between student skills and Job Description
        docs = companies_df.get('Job Description', pd.Series([''] * len(companies_df))).fillna('').astype(str).tolist()
        # Create TF-IDF over job descriptions + student query (fit on docs so vector space matches company descriptions)
        vect = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
        try:
            X = vect.fit_transform(docs)  # (n_companies, n_features)
            q = ' '.join(s_skills)
            qv = vect.transform([q])  # (1, n_features)
            sims = cosine_similarity(qv, X).ravel()  # similarity scores
            for idx, sim in enumerate(sims):
                row = companies_df.iloc[idx]
                matches.append({'company': row.get('Company Name', f"company_{idx}"),
                                'score': float(sim),
                                'overlap': int(sim * 1000),  # heuristic placeholder
                                'avg_salary': float(row.get('avg_salary')) if pd.notna(row.get('avg_salary')) else None})
        except Exception:
            # fallback to set-based Jaccard using keyword extraction (if no TF-IDF)
            # try to build simple keyword sets from job descriptions
            for idx, row in companies_df.iterrows():
                txt = str(row.get('Job Description', '')).lower()
                kws = set([w for w in s_skills if w in txt])
                jaccard = len(kws) / len(set(txt.split())) if txt else 0.0
                matches.append({'company': row.get('Company Name', f"company_{idx}"),
                                'score': float(jaccard),
                                'overlap': int(len(kws)),
                                'avg_salary': float(row.get('avg_salary')) if pd.notna(row.get('avg_salary')) else None})

    # Sort by score desc, tie-break by avg_salary desc
    matches_sorted = sorted(matches, key=lambda x: (x.get('score', 0.0), x.get('avg_salary') or 0.0), reverse=True)
    # Return top_n with cleaned fields (score capped 0..1)
    out = []
    for m in matches_sorted[:top_n]:
        out.append({
            'company': m.get('company'),
            'score': round(float(m.get('score', 0.0)), 4),
            'overlap': int(m.get('overlap', 0)),
            'avg_salary': float(m.get('avg_salary')) if m.get('avg_salary') is not None else None
        })
    return out

def estimate_salary_based_on_progress(matches: List[dict], good_progress: bool):
    """
    If good_progress: return the maximum 'max_salary' available among matches (or avg if max missing).
    Else: return the average 'avg_salary' among matches (or best available fallback).
    """
    if not matches:
        return None
    max_salaries = [m.get("avg_salary") for m in matches if m.get("avg_salary") is not None]
    # some entries might have 'company' only; if the original companies_df had max_salary, we could fetch it.
    if good_progress:
        # prefer max_salary if present in match dict
        max_vals = [m.get("max_salary") for m in matches if m.get("max_salary") is not None]
        if max_vals:
            return max(max_vals)
        # else fallback: top avg
        if max_salaries:
            return max(max_salaries)
    else:
        # return average of avg_salary where available
        if max_salaries:
            return float(np.mean(max_salaries))
    return None


# Streamlit UI
st.title("Student — Sem6 Prediction & Company Match")

students_df = load_students(STUDENT_CSV)
companies_df = load_companies_df(COMPANIES_CSV)
artifact = load_local_model(MODEL_ARTIFACT)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["📊 Predict & Recommend", "💬 Chat with PDFs", "📤 Upload PDFs"])

# Tab 1: Original functionality
with tab1:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Select student")
        if students_df.empty:
            st.warning("No student dataset available. Place simple_student_dataset.csv in repo root.")
            st.stop()
        student_list = students_df['Student_ID'].tolist()
        selected_student = st.selectbox("Student", student_list)
        student_row = students_df[students_df['Student_ID'] == selected_student].iloc[0]

        st.markdown("### Prepopulated semester & personal data (editable)")
        # Build initial form values (use original CSV column names)
        default_payload = {
            "PUC_Score": float(student_row.get("PUC_Score", 0)),
            "Sem_1": float(student_row.get("Sem_1", 0)),
            "Sem_2": float(student_row.get("Sem_2", 0)),
            "Sem_3": float(student_row.get("Sem_3", 0)),
            "Sem_4": float(student_row.get("Sem_4", 0)),
            "Sem_5": float(student_row.get("Sem_5", 0)),
            "Attendance_%": float(student_row.get("Attendance_%", 0)),
            "Study_Hours_Per_Day": float(student_row.get("Study_Hours_Per_Day", 0)),
            "Sleep_Hours_Per_Day": float(student_row.get("Sleep_Hours_Per_Day", 0)),
            "Courses_Completed": str(student_row.get("Courses_Completed", "")),
        }

        # Allow user to choose if prediction should use local model artifact or backend API
        use_api = st.checkbox("Use backend API instead of local model (useful if model artifact not present)", value=False)
        api_url = st.text_input("Backend student_predict URL", value=DEFAULT_API_URL)

        with st.form("student_form"):
            puc = st.number_input("PUC Score", value=default_payload["PUC_Score"], step=0.1)
            sem1 = st.number_input("Sem 1", value=default_payload["Sem_1"], step=0.01)
            sem2 = st.number_input("Sem 2", value=default_payload["Sem_2"], step=0.01)
            sem3 = st.number_input("Sem 3", value=default_payload["Sem_3"], step=0.01)
            sem4 = st.number_input("Sem 4", value=default_payload["Sem_4"], step=0.01)
            sem5 = st.number_input("Sem 5", value=default_payload["Sem_5"], step=0.01)
            attendance = st.number_input("Attendance %", value=default_payload["Attendance_%"], step=0.1)
            study = st.number_input("Study Hours / Day", value=default_payload["Study_Hours_Per_Day"], step=0.1)
            sleep = st.number_input("Sleep Hours / Day", value=default_payload["Sleep_Hours_Per_Day"], step=0.1)
            courses_completed = st.text_input("Courses Completed (comma-separated)", value=default_payload["Courses_Completed"])

            submitted = st.form_submit_button("Predict Sem_6 & Recommend")

    with col2:
        st.subheader("Prediction & Recommendations")
        if 'submitted' in locals() and submitted:
            payload = {
                "PUC_Score": puc,
                "Sem_1": sem1,
                "Sem_2": sem2,
                "Sem_3": sem3,
                "Sem_4": sem4,
                "Sem_5": sem5,
                "Attendance_%": attendance,
                "Study_Hours_Per_Day": study,
                "Sleep_Hours_Per_Day": sleep,
                "Courses_Completed": courses_completed
            }

            predicted_sem6 = None
            api_response = None

            if use_api:
                predicted_sem6, api_response = call_api_predict(api_url, payload)
            else:
                if artifact is None:
                    st.warning("Local model artifact not found. Toggle 'Use backend API' or train the model to create models/sem6_pred.joblib.")
                else:
                    predicted_sem6 = predict_local_sem6(artifact, payload)

            if predicted_sem6 is not None:
                st.success(f"Predicted Sem_6: {predicted_sem6:.3f}")
                # Determine progress: good if predicted >= current Sem_5
                good_progress = predicted_sem6 >= sem5
                st.info("Progress status: " + ("Good (predicted >= Sem_5)" if good_progress else "Needs improvement (predicted < Sem_5)"))

                # company matching
                student_skills = [c.strip().lower() for c in courses_completed.split(',') if c.strip()]
                matches = match_companies_for_student(student_skills, companies_df, top_n=6)
                if matches:
                    # display matches
                    df_matches = pd.DataFrame(matches)
                    # try to enrich with max_salary if companies_df has it
                    if not companies_df.empty and 'max_salary' in companies_df.columns:
                        df_matches['max_salary'] = df_matches['company'].apply(
                            lambda c: float(companies_df.loc[companies_df['Company Name'] == c, 'max_salary'].iloc[0]) if (companies_df.loc[companies_df['Company Name'] == c, 'max_salary'].shape[0] > 0 and pd.notna(companies_df.loc[companies_df['Company Name'] == c, 'max_salary'].iloc[0])) else np.nan
                        )
                    st.markdown("#### Top company matches (by skill overlap):")
                    st.dataframe(df_matches)

                    # estimate salary depending on progress
                    # prefer max_salary if progress good else avg
                    if good_progress:
                        # choose max available max_salary among matches or fallback to avg
                        max_vals = df_matches['max_salary'].dropna().values if 'max_salary' in df_matches.columns else []
                        if len(max_vals) > 0:
                            est_salary = float(np.nanmax(max_vals))
                            st.success(f"Estimated achievable (target) max salary from matched companies: {est_salary:,.2f}")
                        else:
                            avg_vals = df_matches['avg_salary'].dropna().values if 'avg_salary' in df_matches.columns else []
                            if len(avg_vals) > 0:
                                est_salary = float(np.nanmax(avg_vals))
                                st.success(f"Estimated achievable salary (fallback to avg): {est_salary:,.2f}")
                            else:
                                st.info("No salary data available for matched companies.")
                    else:
                        avg_vals = df_matches['avg_salary'].dropna().values if 'avg_salary' in df_matches.columns else []
                        if len(avg_vals) > 0:
                            est_salary = float(np.nanmean(avg_vals))
                            st.info(f"Estimated average target salary based on matches: {est_salary:,.2f}")
                        else:
                            st.info("No salary data available to estimate avg salary.")
                else:
                    st.info("No matching companies found for the student's skills.")
            else:
                st.error("Prediction not available.")

        else:
            st.info("Select a student and click 'Predict Sem_6 & Recommend' to get recommendations.")

    with st.sidebar.form("input_form"):
        st.header("Student Info")
        name = st.text_input("Name", "Alice")
        branch = st.selectbox("Branch", ["Computer Science", "Information Technology", "Electronics", "Unknown"])
        gpa = st.number_input("Current GPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        projects = st.text_area("Projects (short description)", "Project: Image classifier")
        st.markdown("---")
        st.header("Career Goal")
        target_company = st.text_input("Target Company (optional)", "")
        target_ctc = st.number_input("Target CTC (annual, optional)", value=0.0)
        completed_courses = st.text_area("Completed Courses (comma-separated)", "Intro to Python")
        submitted = st.form_submit_button("Ask Coach")

    if submitted:
        payload = {"student_id": 0, "name": name, "branch": branch, "gpa": gpa, "projects": projects}
        try:
            r = requests.post(f"{BACKEND_URL}/student_predict", json=payload, timeout=10)
            pr = r.json()
        except Exception as e:
            pr = {"error": str(e)}
        st.subheader("Predicted Performance")
        if 'predicted_gpa' in pr:
            st.success(f"Predicted GPA (next): {pr['predicted_gpa']:.2f}")
        else:
            st.error(pr.get('error','Prediction failed'))
        rec_payload = {"target_company": target_company if target_company else None,
                       "completed_courses": [c.strip() for c in completed_courses.split(',') if c.strip()],
                       "target_ctc": float(target_ctc) if target_ctc and target_ctc>0 else None}
        try:
            rr = requests.post(f"{BACKEND_URL}/student_recommendations", json=rec_payload, timeout=15)
            rec = rr.json()
        except Exception as e:
            rec = {"error": str(e)}
        st.subheader("Course Recommendations")
        if 'recommendations' in rec:
            for r in rec['recommendations']:
                st.markdown(f"**{r.get('course_name')}** — skills: {r.get('skills')} — score: {r.get('score')}")
                if r.get('url'):
                    st.markdown(f"[Course link]({r.get('url')})")
        else:
            st.info(rec.get('error', 'No recommendations available.'))

# Tab 2: Chat interface
with tab2:
    st.header("💬 Chat with Your Documents")
    st.markdown("Ask questions about uploaded PDFs and get course recommendations!")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents or courses..."):
        print("Send chat request :== ", prompt)
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from backend
        try:
            print("Send chat request : ", prompt)
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"query": prompt, "top_k": 5},
                timeout=30
            )
            result = response.json()

            if "error" in result:
                answer = f"Error: {result['error']}"
            else:
                answer = result.get("answer", "No answer available")

                # Show additional info in expandable sections
                if result.get("pdf_sources"):
                    answer += f"\n\n📄 **Sources**: {', '.join(set(result['pdf_sources']))}"

                if result.get("recommended_courses"):
                    answer += f"\n\n📚 **Recommended Courses**: {', '.join(result['recommended_courses'])}"
        except Exception as e:
            answer = f"Error connecting to backend: {str(e)}"

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# Tab 3: Upload PDFs
with tab3:
    st.header("📤 Upload PDFs for Indexing")
    st.markdown("Upload course materials, textbooks, or any educational PDFs to enable semantic search and chat.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Upload and Index"):
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload_pdf", files=files, timeout=60)
                    result = response.json()

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Indexed {result['chunks_indexed']} chunks from {result['filename']}")
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")

    st.markdown("---")
    st.markdown("### Search Documents")
    search_query = st.text_input("Search for specific topics...")
    if search_query:
        try:
            response = requests.post(
                f"{BACKEND_URL}/search",
                json={"query": search_query, "top_k": 5},
                timeout=15
            )
            result = response.json()

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.write(f"Found {result['total_results']} results:")
                for item in result.get("results", []):
                    with st.expander(f"#{item['rank']} - {item['source']} (score: {item['score']:.4f})"):
                        st.write(item['text'])
        except Exception as e:
            st.error(f"Error searching: {str(e)}")

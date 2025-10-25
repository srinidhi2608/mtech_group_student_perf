import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Student Coach Chatbot", layout="wide")
st.title("Student Coach — Predict & Recommend")

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
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
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
        rr = requests.post(f"{BACKEND_URL}/recommend", json=rec_payload, timeout=15)
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

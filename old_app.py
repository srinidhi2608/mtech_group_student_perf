import os
import streamlit as st
import pandas as pd

from chatbot import ChatBot
from data_processing import load_students_csv, load_predictions_csv
from syllabus_parser import parse_pdf_to_text

st.set_page_config(page_title="Student CTC Chatbot", layout="wide")

st.title("AI Chatbot — Course Recommendations to Improve CTC")

# Sidebar: upload files
st.sidebar.header("Uploads & Settings")
students_file = st.sidebar.file_uploader("Upload students CSV", type=["csv"])
predictions_file = st.sidebar.file_uploader("Upload predictions CSV", type=["csv"])
syllabus_files = st.sidebar.file_uploader("Upload syllabus PDF(s)", type=["pdf"], accept_multiple_files=True)

openai_key = os.getenv("OPENAI_API_KEY", "")
use_openai = st.sidebar.checkbox("Use OpenAI for responses (optional)", value=bool(openai_key))
if use_openai and not openai_key:
    st.sidebar.info("Set OPENAI_API_KEY in env to enable OpenAI")

# Control: choose student
student_df = None
pred_df = None
if students_file:
    student_df = load_students_csv(students_file)
if predictions_file:
    pred_df = load_predictions_csv(predictions_file)

student_id = None
if student_df is not None:
    st.sidebar.subheader("Select Student")
    # assume a column 'student_id' or 'id' or 'name'
    id_cols = []
    for c in ["student_id", "id", "studentID", "roll", "name"]:
        if c in student_df.columns:
            id_cols.append(c)
    if not id_cols:
        st.sidebar.warning("students CSV missing an obvious id column. Using first column.")
        id_cols = [student_df.columns[0]]
    display_col = id_cols[0]
    options = student_df[display_col].astype(str).tolist()
    selected = st.sidebar.selectbox("Choose", options)
    student_id = selected

# Parse syllabus PDFs
syllabus_texts = {}
if syllabus_files:
    st.sidebar.write(f"Parsing {len(syllabus_files)} syllabus PDF(s)...")
    for f in syllabus_files:
        try:
            text = parse_pdf_to_text(f)
            syllabus_texts[f.name] = text
        except Exception as e:
            st.sidebar.error(f"Error parsing {f.name}: {e}")

# Initialize chatbot
bot = ChatBot(student_df=student_df, pred_df=pred_df, syllabus_texts=syllabus_texts, use_openai=use_openai)

# Main chat UI
st.subheader("Chat")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are an assistant that helps students improve CTC by recommending courses and explaining predictions."}]

# Display chat history
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Input box
def submit():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return
    st.session_state.messages.append({"role": "user", "content": user_text})
    # generate reply
    reply = bot.respond(user_text, student_identifier=student_id)
    st.session_state.messages.append({"role": "bot", "content": reply})

st.text_input("Type your question here...", key="user_input", on_change=submit)
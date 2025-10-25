import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Student Coach Chatbot", layout="wide")
st.title("🎓 Student Coach — Predict, Recommend & Chat")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["📊 Predict & Recommend", "💬 Chat with PDFs", "📤 Upload PDFs"])

# Tab 1: Original functionality
with tab1:
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
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        try:
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

# AI Course-Recommendation Chatbot (Student CTC Improvement)

This project is a Python-based AI chatbot with a web UI (Streamlit) designed to help students improve their campus-selection CTC (salary) by:
- Using a student dataset
- Using a syllabus PDF (or multiple syllabus PDFs) to see what courses cover which skills
- Incorporating existing student performance prediction results (CSV with predicted CTC)
- Suggesting courses that may increase predicted CTC
- Answering conversational questions about students, recommendations, and the syllabus

Key features
- Simple chat UI for conversational queries
- Student selection and context-aware answers (uses the student record + predicted CTC)
- Syllabus PDF parsing and lightweight retrieval (keyword matching)
- Rule-based recommender with optional LLM-enhanced replies (OpenAI support is optional)

Quick demo (local)
1. Install dependencies:
   - pip install -r requirements.txt

2. Run the app:
   - streamlit run app.py

3. Upload:
   - students.csv (student dataset)
   - predictions.csv (prediction results with predicted_ctc column)
   - syllabus PDF(s)

4. Choose a student in the sidebar and chat using the chat UI. Ask things like:
   - "Which courses from the uploaded syllabus would help increase my CTC?"
   - "What's my predicted CTC?"
   - "What skills should I focus on to improve my chances in campus placements?"

OpenAI (optional)
- The app has optional OpenAI integration for nicer conversational replies.
- Set environment variable OPENAI_API_KEY to enable.
- If not set, the chatbot uses a robust rule-based fallback.

File list
- app.py: Streamlit UI
- chatbot.py: Chatbot orchestration (context building + LLM or fallback)
- data_processing.py: CSV loaders and record helpers
- syllabus_parser.py: Extract text from PDF and build simple index
- recommender.py: Course suggestion logic
- requirements.txt
- students_sample.csv, predictions_sample.csv: example inputs
- .gitignore

Notes & Next steps
- Swap in better NLP pipelines (embeddings + vector DB) for more advanced retrieval.
- Replace rule-based recommender with an ML model trained on historical uplift data.
- Add scheduling (course timelines), course metadata (duration, cost) and post-course uplift simulation.

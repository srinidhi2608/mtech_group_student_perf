import os
import openai
import textwrap
from typing import Optional

from data_processing import get_student_record, get_predicted_ctc
from recommender import suggest_courses_for_student

OPENAI_MODEL = "gpt-3.5-turbo"

class ChatBot:
    def __init__(self, student_df=None, pred_df=None, syllabus_texts=None, use_openai: bool = False):
        self.student_df = student_df
        self.pred_df = pred_df
        self.syllabus_texts = syllabus_texts or {}
        self.use_openai = use_openai and bool(os.getenv("OPENAI_API_KEY"))
        if self.use_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def build_context(self, student_identifier: Optional[str]):
        ctx_parts = []
        if student_identifier and self.student_df is not None:
            rec = get_student_record(self.student_df, student_identifier)
            if rec is not None:
                ctx_parts.append("Student record:")
                for k, v in rec.items():
                    ctx_parts.append(f"- {k}: {v}")
        if self.pred_df is not None and student_identifier:
            pred = get_predicted_ctc(self.pred_df, student_identifier)
            if pred is not None:
                ctx_parts.append(f"Predicted CTC: {pred} (from prediction CSV)")
        if self.syllabus_texts:
            ctx_parts.append("Syllabus summary: available. Use when matching course content to needed skills.")
        return "\n".join(ctx_parts)

    def respond(self, user_message: str, student_identifier: Optional[str] = None) -> str:
        # Compose context
        context = self.build_context(student_identifier)
        # If the user's message asks for recommendations, call recommender directly
        lower = user_message.lower()
        if any(k in lower for k in ["recommend", "suggest", "courses", "improve", "ctc", "salary"]):
            # Use recommender
            recs = suggest_courses_for_student(self.student_df, self.pred_df, student_identifier, self.syllabus_texts)
            reply = "Here are recommended courses to improve predicted CTC:\n"
            if not recs:
                reply += "No strong suggestions found. Upload a predictions CSV and a syllabus PDF to enable better suggestions."
            else:
                for r in recs:
                    reply += f"- {r['course_title']}: matches skills {r['matched_skills']}. Estimated uplift: {r.get('estimated_uplift','~unknown')}\n"
            # Optionally expand with LLM for nicer phrasing
            if self.use_openai:
                prompt = f"""
You are an assistant that gets the following context:
{context}

User asked:
{user_message}

Raw recommendations:
{reply}

Please produce a concise friendly reply summarizing the recommendations and giving next steps for a student to follow.
"""
                try:
                    resp = openai.ChatCompletion.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "system", "content": "You are a helpful academic/career advisor."},
                                  {"role": "user", "content": prompt}],
                        max_tokens=400
                    )
                    content = resp["choices"][0]["message"]["content"].strip()
                    return content
                except Exception as e:
                    # fallback
                    return reply + f"\n(Note: OpenAI call failed: {e})"
            return reply

        # For general chat: call LLM if available, else build rule-based answer
        if self.use_openai:
            prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
            try:
                resp = openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": "You are a helpful assistant advising students on courses and CTC improvement."},
                              {"role": "user", "content": prompt}],
                    max_tokens=400
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return f"Error calling OpenAI: {e}\n\nContext:\n{context}"
        else:
            # fallback rule-based
            if "predicted" in lower and student_identifier:
                pred = get_predicted_ctc(self.pred_df, student_identifier)
                if pred is not None:
                    return f"The predicted CTC for the selected student is: {pred} (from predictions CSV)."
                else:
                    return "I don't have prediction data for that student. Upload predictions.csv with a 'student_id' and 'predicted_ctc' column."
            return textwrap.dedent(f"""
            I can help with:
            - Suggesting courses to improve predicted CTC (ask 'recommend courses' or 'suggest courses').
            - Showing predicted CTC (ask 'what is my predicted CTC').
            Current context:
            {context}
            """).strip()
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


def compare_ai_with_human(
    model_text,
    student_text,
    ai_score,
    ai_breakdown,
    human_score,
    human_feedback
):
    prompt = f"""
You are an academic moderation system.
You are given:
- Model answer
- Student answer
- AI grading result with topic-wise breakdown
- Human examiner score and feedback

Your task:
1. Judge who is more justified
2. Identify agreement and disagreement
3. State where AI did better than human
4. State where human judgment was superior
5. Give a final verdict on AI reliability
6. Your score to student
7. Rate the AI grading quality out of 10

---

MODEL ANSWER:
{model_text}

---

STUDENT ANSWER:
{student_text}

---

AI SCORE:
{ai_score}

AI BREAKDOWN (JSON):
{json.dumps(ai_breakdown, indent=2)}

---

HUMAN SCORE:
{human_score}

HUMAN FEEDBACK (JSON):
{json.dumps(human_feedback, indent=2)}

---

Return a structured, clear academic analysis.
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
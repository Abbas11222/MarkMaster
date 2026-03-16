import os
import json
from config import groq_client as client, LLM_MODEL as MODEL_NAME


def compare_ai_with_human(
    model_text,
    student_text,
    ai_score,
    ai_breakdown,
    human_score,
    human_feedback
):
    total_marks = sum(
        item.get("marks_available", 0)
        for item in ai_breakdown
        if item.get("topic") == "Overall Answer"
    ) or max(
        (item.get("marks_available", 0) for item in ai_breakdown), default=10
    )

    prompt = f"""
You are an academic moderation system.
You are given:
- Model answer
- Student answer
- AI grading result with topic-wise breakdown
- Human examiner score and feedback

Your task:
1. Based strictly on the model answer and student answer content, determine which score is more accurate — the AI score or the human score. Support your decision with specific evidence only.
2. Identify agreement and disagreement
3. State where AI did better than human
4. State where human judgment was superior
5. Give a final verdict on AI reliability
6. Rate the AI grading quality out of 10

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

Return a structured, evidence-based academic analysis.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0   # zero temperature = no creativity, pure analysis
    )

    return response.choices[0].message.content
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


def llm_validate_score(model_text, student_text, ai_score, max_marks, topic):
    """
    Uses LLM to validate and potentially adjust the AI-computed score.
    Returns a dict with adjusted_score, justification, and verdict.
    """

    prompt = f"""
You are a strict but fair academic examiner.

Topic Being Graded: {topic}
Maximum Marks Available: {max_marks}
Preliminary AI Score: {ai_score}

Model Answer (what the student should have written):
{model_text}

Student Answer (what the student actually wrote):
{student_text}

Your tasks:
1. Check if the student actually addressed the topic correctly
2. Check how well the student covered key concepts from the model answer
3. Check for factual accuracy in the student answer
4. Decide if the AI preliminary score is fair, too generous, or too strict
5. Give an adjusted score that reflects the true quality of the answer

Return ONLY a valid JSON object with no extra text, no markdown, no explanation outside the JSON:
{{
  "adjusted_score": <float between 0 and {max_marks}>,
  "justification": "<one clear sentence explaining your decision>",
  "verdict": "<one of: fair | too_generous | too_strict>"
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        # Safely extract JSON even if there's surrounding text
        start = content.find("{")
        end = content.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")

        parsed = json.loads(content[start:end])

        # Clamp the score to valid range
        parsed["adjusted_score"] = max(0.0, min(float(max_marks), float(parsed["adjusted_score"])))

        return parsed

    except Exception as e:
        print(f"[llm_scorer] Warning: LLM validation failed for topic '{topic}': {e}")
        return {
            "adjusted_score": float(ai_score),
            "justification": "LLM validation failed, using original AI score.",
            "verdict": "fair"
        }
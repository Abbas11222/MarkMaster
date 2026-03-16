"""
rag_answer_builder.py
─────────────────────
Given a question + retrieved chunks, uses LLM to synthesize
a structured model answer in the same format as text_extracr.py output.

Output format matches exactly what matcher.py expects:
{
  "questions": [
    {
      "question_id": "Q1",
      "question_title": "...",
      "parts": [
        {
          "part_id": "main",
          "topic": "...",
          "content": "",
          "sub_topics": [
            { "topic": "...", "content": "...", "sub_topics": [] }
          ]
        }
      ]
    }
  ]
}
"""

import os
import json
from vector_store import retrieve
from config import groq_client as client, LLM_MODEL as MODEL_NAME


def build_model_answer(question_id, question_text, top_k=6):
    """
    Main function: given a question, retrieve relevant chunks
    and synthesize a structured model answer.

    Returns the standard extracted paper format:
    { "questions": [...], "source": "RAG" }
    """
    print(f"\n  🔍 Retrieving chunks for {question_id}...")
    chunks = retrieve(question_text, top_k=top_k)

    if not chunks:
        print(f"  ⚠️ No relevant chunks found for {question_id}")
        return None

    # Format chunks for the prompt
    context_block = ""
    for i, c in enumerate(chunks, 1):
        context_block += f"\n[Source {i}: {c['source']}, page {c['page']} — relevance {c['score']}]\n{c['text']}\n"

    prompt = f"""You are an academic exam answer generator.

Using ONLY the course material provided below, write a complete model answer
for the following exam question. Do not use any outside knowledge.

EXAM QUESTION:
{question_text}

COURSE MATERIAL (retrieved relevant sections):
{context_block}

Generate a structured model answer and return it as JSON in EXACTLY this format:
{{
  "question_id": "{question_id}",
  "question_title": "<short title summarizing the question>",
  "topics": [
    {{ "heading": "<topic heading>", "content": "<detailed explanation>" }},
    {{ "heading": "<topic heading>", "content": "<detailed explanation>" }}
  ]
}}

RULES:
1. topics array must have at least 2 entries — one per major point the question asks about
2. heading must never be empty — use a clear label for each topic
3. content must be detailed — at least 2-3 sentences per topic
4. Base content ONLY on the provided course material
5. Return ONLY the JSON — no markdown, no extra text
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw  = json.loads(res.choices[0].message.content)
        return _normalize_rag_output(raw, question_id, chunks)

    except Exception as e:
        print(f"  ❌ RAG answer generation failed for {question_id}: {e}")
        return None


def _normalize_rag_output(raw, question_id, source_chunks):
    """
    Convert RAG LLM output into the standard format
    that matcher.py and component_builder.py expect.
    """
    qid    = raw.get("question_id", question_id)
    qtitle = raw.get("question_title", "")
    topics = raw.get("topics", [])

    sub_topics = []
    for t in topics:
        heading = str(t.get("heading", "")).strip()
        content = str(t.get("content", "")).strip()
        if not heading and not content:
            continue
        sub_topics.append({
            "topic":      heading,
            "content":    content,
            "sub_topics": []
        })

    if not sub_topics:
        return None

    # Include source attribution metadata
    sources = list({c["source"] for c in source_chunks})

    return {
        "questions": [{
            "question_id":    qid,
            "question_title": qtitle,
            "parts": [{
                "part_id":    "main",
                "topic":      qtitle,
                "content":    "",
                "sub_topics": sub_topics
            }]
        }],
        "source":       "RAG",
        "source_files": sources
    }


def build_all_model_answers(questions_dict):
    """
    Build model answers for multiple questions at once.

    questions_dict = { "Q1": "Explain formal methods...", "Q2": "..." }

    Returns list of paper dicts (same format as process_folder output).
    """
    papers = []
    for qid, qtext in questions_dict.items():
        result = build_model_answer(qid, qtext)
        if result:
            papers.append(result)
            print(f"  ✅ Model answer built for {qid} "
                  f"({len(result['questions'][0]['parts'][0]['sub_topics'])} topics)")
        else:
            print(f"  ⚠️ Skipped {qid} — no answer generated")
    return papers
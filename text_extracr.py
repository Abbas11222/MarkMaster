import os
import base64
import json
import glob
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def normalize_qid(qid):
    numbers = re.findall(r'\d+', str(qid))
    if numbers:
        return f"Q{int(numbers[0])}"
    return "Q1"


# =====================================================
# PASS 1 — Find question label positions
# Simple task: just find labels and their positions
# =====================================================

PASS1_PROMPT = """
Look at this handwritten exam paper image.
Find all question number labels (e.g. "Answer 04", "Q3", "Question 2").

Return ONLY this JSON:
{
  "labels": [
    { "question_id": "Q4", "position": "top" }
  ]
}

position = "top" if in upper half, "bottom" if in lower half.
If NO label found → return: { "labels": [] }
"""


# =====================================================
# PASS 2 — Extract content with boundary context
# =====================================================

def build_pass2_prompt(labels):
    if not labels:
        boundary_instruction = """
This page has NO question label. Use question_id "CONTINUATION" for ALL content.
"""
    elif len(labels) == 1:
        qid = labels[0]["question_id"]
        pos = labels[0]["position"]
        if pos == "bottom":
            boundary_instruction = f"""
Label "{qid}" is in the BOTTOM half.
- Content ABOVE the label → question_id "CONTINUATION"
- Content FROM label downward → question_id "{qid}"
Create TWO objects if there is content above the label, otherwise ONE.
"""
        else:
            boundary_instruction = f"""
Label "{qid}" is at the TOP. All content belongs to {qid}.
Create ONE question object.
"""
    else:
        sorted_labels = sorted(labels, key=lambda x: 0 if x["position"] == "top" else 1)
        desc = "\n".join(f'  - "{l["question_id"]}" at {l["position"]}' for l in sorted_labels)
        qids = [l["question_id"] for l in sorted_labels]
        boundary_instruction = f"""
Multiple labels found:
{desc}
Assign content to the question it appears UNDER.
Content before first label → "CONTINUATION".
Create separate objects for: {", ".join(qids)}.
"""

    return f"""
Extract handwritten academic answers from this exam image.

PAGE STRUCTURE:
{boundary_instruction}

OUTPUT FORMAT:
{{
  "questions": [
    {{
      "question_id": "Q2",
      "question_title": "",
      "topics": [
        {{ "heading": "", "content": "" }}
      ]
    }}
  ]
}}

RULES:
1. topics = flat array, never nested
2. All text under a heading → ONE content string
3. Extract verbatim, do not paraphrase
4. Return ONLY the JSON, no markdown
5. topics never empty — use heading:"" if no headings exist
"""


def normalize_output(data):
    if not isinstance(data, dict):
        return None
    questions_raw = data.get("questions", [])
    if not isinstance(questions_raw, list) or not questions_raw:
        return None

    questions_out = []
    for q in questions_raw:
        if not isinstance(q, dict):
            continue
        raw_qid = q.get("question_id", "Q1")
        qid     = "CONTINUATION" if str(raw_qid).strip().upper() == "CONTINUATION" \
                  else normalize_qid(raw_qid)
        qtitle  = str(q.get("question_title", "")).strip()
        topics  = q.get("topics", [])
        if not isinstance(topics, list):
            topics = []

        sub_topics = []
        for t in topics:
            if not isinstance(t, dict):
                continue
            heading = str(t.get("heading", "")).strip()
            content = str(t.get("content", "")).strip()
            if not heading and not content:
                continue
            sub_topics.append({"topic": heading, "content": content, "sub_topics": []})

        if not sub_topics:
            continue

        questions_out.append({
            "question_id":    qid,
            "question_title": qtitle,
            "parts": [{
                "part_id":    "main",
                "topic":      qtitle,
                "content":    "",
                "sub_topics": sub_topics
            }]
        })

    return {"questions": questions_out} if questions_out else None


# =====================================================
# TWO-PASS EXTRACTION (with fast-path for simple pages)
# =====================================================

def extract_content_from_image(image_path, max_retries=3):
    encoded  = encode_image(image_path)
    filename = os.path.basename(image_path)

    # ── PASS 1: Find labels ───────────────────────────
    labels = []
    try:
        res1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [
                {"type": "text",      "text": PASS1_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]}],
            response_format={"type": "json_object"},
            seed=42, temperature=0.0, top_p=1
        )
        pass1_data = json.loads(res1.choices[0].message.content)
        labels = pass1_data.get("labels", [])
        for label in labels:
            raw = label.get("question_id", "")
            if raw and str(raw).strip().upper() != "CONTINUATION":
                label["question_id"] = normalize_qid(raw)
        print(f"  📍 {filename}: {[l['question_id']+'@'+l['position'] for l in labels]}")

    except Exception as e:
        print(f"  ⚠️ Pass 1 failed for {filename}: {e}")
        labels = []

    # ── PASS 2: Extract content ───────────────────────
    pass2_prompt = build_pass2_prompt(labels)

    for attempt in range(1, max_retries + 1):
        try:
            res2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    {"type": "text",      "text": pass2_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ]}],
                response_format={"type": "json_object"},
                seed=42, temperature=0.0, top_p=1
            )
            raw        = json.loads(res2.choices[0].message.content)
            normalized = normalize_output(raw)
            if normalized:
                return normalized
            raise ValueError("Normalization failed")

        except Exception as e:
            print(f"  ⚠️ Pass 2 attempt {attempt}/{max_retries} failed for {filename}: {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                print(f"  ❌ All attempts failed for {filename}")
                return None


# =====================================================
# PROCESS FOLDER — parallel processing
# Processes all images concurrently instead of one by one
# For 3 images: was 3×(2 API calls + delays) = ~30s sequential
#               now all 3 run at same time = ~12s parallel
# =====================================================

def process_single(file_path):
    """Process one image — used by thread pool."""
    print(f"-> Processing: {os.path.basename(file_path)}")
    data = extract_content_from_image(file_path)
    if data:
        data["source"] = os.path.basename(file_path)
        return file_path, data
    else:
        print(f"  ⚠️ Skipped {os.path.basename(file_path)}")
        return file_path, None


def process_folder(folder_path, max_workers=3):
    """
    Processes all images in parallel.
    max_workers=3 means up to 3 images processed simultaneously.
    Results are sorted by filename to preserve page order.
    """
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder_path, p), recursive=True))
    files = sorted(files)

    if not files:
        return []

    results_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, f): f for f in files}
        for future in as_completed(futures):
            file_path, data = future.result()
            if data:
                results_map[file_path] = data

    # Return in sorted order (page order)
    return [results_map[f] for f in files if f in results_map]
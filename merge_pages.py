"""
merge_pages.py

Key rule: pages are uploaded in sequence.
When a page has TWO questions (end of Q2 + start of Q3):
  - Content BEFORE the new label  → appended to PREVIOUS question (CONTINUATION)
  - Content FROM the new label    → goes into new question bucket

text_extracr.py handles this split via Pass 1 (find label positions) +
Pass 2 (boundary-aware extraction), producing blocks labelled either
CONTINUATION or the actual question ID.
"""

import re


def normalize_qid(qid: str) -> str:
    numbers = re.findall(r'\d+', str(qid))
    return f"Q{int(numbers[0])}" if numbers else str(qid).strip().upper()


def _has_content(parts: list) -> bool:
    for part in parts:
        if not isinstance(part, dict):
            continue
        for st in part.get("sub_topics", []):
            if isinstance(st, dict):
                if st.get("topic", "").strip() or st.get("content", "").strip():
                    return True
    return False


def _merge_parts_into(target_q: dict, new_parts: list):
    for part in new_parts:
        if not isinstance(part, dict):
            continue
        sub_topics = part.get("sub_topics", [])
        if not isinstance(sub_topics, list):
            continue
        if not sub_topics and not part.get("content", "").strip():
            continue
        part_id  = part.get("part_id", "main")
        existing = next(
            (p for p in target_q["parts"] if p.get("part_id") == part_id), None
        )
        if existing is None:
            target_q["parts"].append({
                "part_id":    part_id,
                "topic":      part.get("topic", ""),
                "content":    part.get("content", ""),
                "sub_topics": list(sub_topics)
            })
        else:
            existing["sub_topics"].extend(sub_topics)


def _new_bucket(qid: str, title: str = "") -> dict:
    return {"question_id": qid, "question_title": title, "parts": []}


def merge_extractions(extracted_pages: list, debug: bool = True) -> list:
    """
    Merge per-page extractions into one bucket per question.

    Three page patterns handled:
      A) Full page = one question             → simple merge into bucket
      B) Full page = no label (continuation)  → append to previous question
      C) Mid-page boundary (two questions)    → text_extracr already split it:
            CONTINUATION block  → appended to prev question
            New Q label block   → new question bucket
    """
    merged     = {}   # qid → bucket dict
    page_order = []   # insertion order for output
    prev_qid   = None

    def ensure_bucket(qid, title=""):
        if qid not in merged:
            merged[qid] = _new_bucket(qid, title)
            page_order.append(qid)
            if debug:
                print(f"    [bucket] Created '{qid}'")
        else:
            if not merged[qid]["question_title"] and title:
                merged[qid]["question_title"] = title

    for page_idx, page in enumerate(extracted_pages):
        if not isinstance(page, dict):
            continue

        source    = page.get("source", f"page_{page_idx}")
        questions = page.get("questions", [])

        if not isinstance(questions, list) or not questions:
            if debug:
                print(f"[merge] {source}: no questions — skip")
            continue

        if debug:
            ids = [q.get("question_id", "?") for q in questions]
            print(f"\n[merge] {source}: ids={ids}  prev_qid={prev_qid}")

        for q in questions:
            if not isinstance(q, dict):
                continue

            raw_id = str(q.get("question_id", "Q1")).strip().upper()
            title  = q.get("question_title", "").strip()
            parts  = q.get("parts", []) if isinstance(q.get("parts"), list) else []

            # ── CASE 1: CONTINUATION ──────────────────────────────
            # text_extracr labels content ABOVE a new question label
            # as CONTINUATION. It belongs to the previous question.
            if raw_id == "CONTINUATION":
                target = prev_qid if prev_qid else "Q1"
                if debug:
                    print(f"  CONTINUATION → appending to '{target}'")
                ensure_bucket(target)
                _merge_parts_into(merged[target], parts)
                # Do NOT change prev_qid — CONTINUATION is not a new question

            # ── CASE 2: Q1 default fallback (no label found) ─────
            # Extractor defaults to Q1 when it finds no question label.
            # If we already know prev_qid, this is just more content
            # for the current question.
            elif raw_id == "Q1" and not title and prev_qid and prev_qid != "Q1":
                target = prev_qid
                if debug:
                    print(f"  Q1(default, no title) → appending to '{target}'")
                ensure_bucket(target)
                _merge_parts_into(merged[target], parts)
                # Do NOT change prev_qid

            # ── CASE 3: Real explicit question label ──────────────
            # Q2, Q3, Question 4, etc. → new or existing bucket.
            # Always update prev_qid so future pages know where we are.
            else:
                target   = normalize_qid(raw_id)
                prev_qid = target          # update FIRST so CONTINUATION on
                                           # the NEXT page knows this question
                if debug:
                    print(f"  {raw_id} → '{target}'  (prev_qid updated)")
                ensure_bucket(target, title)
                _merge_parts_into(merged[target], parts)

    # ── Debug summary ─────────────────────────────────────────────
    if debug:
        print(f"\n[merge] Final buckets:")
        for qid in page_order:
            total_st = sum(
                len(p.get("sub_topics", []))
                for p in merged[qid]["parts"]
                if isinstance(p, dict)
            )
            print(f"  {qid}: {len(merged[qid]['parts'])} parts, {total_st} sub_topics")

    # ── Build output — skip empty buckets ─────────────────────────
    result = []
    for qid in page_order:
        bucket = merged[qid]
        if not _has_content(bucket["parts"]):
            if debug:
                print(f"[merge] WARNING: '{qid}' has no content — skipped")
            continue
        result.append({"questions": [bucket]})

    return result



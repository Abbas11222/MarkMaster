"""
merge_pages.py — with debug tracing to find where Q3 disappears.
"""

import re


def normalize_qid(qid):
    numbers = re.findall(r'\d+', str(qid))
    if numbers:
        return f"Q{int(numbers[0])}"
    return str(qid).strip().upper()


def is_default_fallback(qid):
    return str(qid).strip().upper() in ("Q1", "CONTINUATION")


def merge_parts_into(target_q, new_parts):
    for part in new_parts:
        if not isinstance(part, dict):
            continue
        sub_topics = part.get("sub_topics", [])
        if not isinstance(sub_topics, list):
            continue
        if not sub_topics and not part.get("content", "").strip():
            continue

        existing = None
        for ep in target_q["parts"]:
            if ep.get("part_id") == part.get("part_id", "main"):
                existing = ep
                break

        if existing is None:
            target_q["parts"].append({
                "part_id":    part.get("part_id", "main"),
                "topic":      part.get("topic", ""),
                "content":    part.get("content", ""),
                "sub_topics": list(sub_topics)
            })
        else:
            existing["sub_topics"].extend(sub_topics)


def merge_extractions(extracted_pages, debug=True):
    """
    Merges per-page extractions into one dict per question.
    debug=True prints a trace of every decision made.
    """
    merged     = {}
    page_order = []
    prev_qid   = None

    for page_idx, page in enumerate(extracted_pages):
        if not isinstance(page, dict):
            continue

        source    = page.get("source", f"page_{page_idx}")
        questions = page.get("questions", [])

        if not isinstance(questions, list) or not questions:
            if debug:
                print(f"  [merge] {source}: no questions found — skipping")
            continue

        if debug:
            qids_on_page = [q.get("question_id", "?") for q in questions]
            print(f"  [merge] {source}: questions on page = {qids_on_page}, prev_qid = {prev_qid}")

        # ── Detect pure continuation page ──────────────────────────
        if len(questions) == 1:
            q      = questions[0]
            raw_id = str(q.get("question_id", "Q1")).strip().upper()
            title  = q.get("question_title", "").strip()

            if raw_id in ("CONTINUATION", "Q1") and not title and prev_qid:
                if debug:
                    print(f"    → {source}: single default page, reassigning to prev_qid={prev_qid}")
                questions[0]["question_id"] = prev_qid

        # ── Process each question on this page ──────────────────────
        for q in questions:
            if not isinstance(q, dict):
                continue

            raw_id = str(q.get("question_id", "Q1")).strip().upper()
            title  = q.get("question_title", "").strip()
            parts  = q.get("parts", [])

            # Count sub_topics to know if there's real content
            total_subtopics = sum(
                len(p.get("sub_topics", [])) for p in parts
                if isinstance(p, dict)
            )

            # ── Resolve target question ID ─────────────────────────
            if raw_id == "CONTINUATION":
                target = prev_qid if prev_qid else "Q1"
                if debug:
                    print(f"    → CONTINUATION → assigned to {target} ({total_subtopics} subtopics)")

            elif raw_id == "Q1" and not title and prev_qid and prev_qid != "Q1":
                target = prev_qid
                if debug:
                    print(f"    → Q1 (default, no title) → assigned to prev_qid={target} ({total_subtopics} subtopics)")

            else:
                target = normalize_qid(raw_id)
                if debug:
                    print(f"    → {raw_id} → normalized to {target} ({total_subtopics} subtopics)")

            # ── Initialize or update bucket ────────────────────────
            if target not in merged:
                merged[target] = {
                    "question_id":    target,
                    "question_title": title or "",
                    "parts":          []
                }
                page_order.append(target)
                if debug:
                    print(f"    → Created new bucket for {target}")
            else:
                if not merged[target]["question_title"] and title:
                    merged[target]["question_title"] = title
                if debug:
                    print(f"    → Merging into existing bucket {target}")

            # ── Merge content ──────────────────────────────────────
            merge_parts_into(merged[target], parts)

            # ── Update prev_qid ────────────────────────────────────
            # Only update for real explicit IDs (not CONTINUATION or default Q1)
            if raw_id not in ("CONTINUATION",):
                real = normalize_qid(raw_id) if raw_id != "Q1" else "Q1"
                if raw_id != "Q1" or title:
                    prev_qid = real
                    if debug:
                        print(f"    → prev_qid updated to {prev_qid}")

    # ── Summary ────────────────────────────────────────────────────
    if debug:
        print(f"\n  [merge] Final buckets:")
        for qid in page_order:
            total = sum(
                len(p.get("sub_topics", []))
                for p in merged[qid]["parts"]
            )
            print(f"    {qid}: {total} subtopics, parts={len(merged[qid]['parts'])}")

    # ── Build output ───────────────────────────────────────────────
    result = []
    for qid in page_order:
        q_data = merged[qid]
        if not q_data["parts"]:
            if debug:
                print(f"  [merge] WARNING: {qid} has no parts — skipping")
            continue
        result.append({"questions": [q_data]})

    return result
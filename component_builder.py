def collect_full_text(topic):
    """Recursively collect all text from topic and descendants."""
    texts = []

    def recurse(node):
        topic_name = node.get("topic", "")
        content    = node.get("content", "")
        combined   = f"{topic_name}. {content}".strip(". ").strip()
        if combined:
            texts.append(combined)
        for sub in node.get("sub_topics", []):
            recurse(sub)

    recurse(topic)
    return " ".join(texts)


def collect_content_only(topic):
    """
    Recursively collect ONLY content text (no topic/heading names).
    Used for holistic scoring — avoids heading name mismatches
    like "(a). Ghost variables" vs "Ghost variables" scoring differently.
    """
    texts = []

    def recurse(node):
        content = node.get("content", "").strip()
        if content:
            texts.append(content)
        for sub in node.get("sub_topics", []):
            recurse(sub)

    recurse(topic)
    return " ".join(texts)


def flatten_topic(topic, parent_id, weight, components, question_id, part_id,
                  question_context="", main_ratio=0.3):

    subs       = topic.get("sub_topics", [])
    full_text  = collect_full_text(topic)
    topic_name = topic.get("topic", "").strip()

    # content_only = pure answer content, no headings (for holistic scoring)
    # raw_text     = heading + content (for component BERTScore without prefix)
    # text         = context-enriched (for component BERTScore with question context)
    content_only = collect_content_only(topic)
    raw_text     = f"{topic_name}. {full_text}".strip(". ").strip()

    if question_context:
        enriched_text = f"{question_context}. {raw_text}"
    else:
        enriched_text = raw_text

    if not subs:
        if raw_text or topic_name:
            components.append({
                "id":           parent_id,
                "question":     question_id,
                "part":         part_id,
                "topic":        topic_name,
                "text":         enriched_text,   # component-level BERTScore
                "raw_text":     raw_text,         # fallback
                "content_only": content_only,     # holistic scoring
                "weight":       float(weight),
                "is_critical":  False
            })
        return

    main_weight = weight * main_ratio
    remaining   = weight * (1 - main_ratio)
    sub_weight  = remaining / len(subs)

    if raw_text or topic_name:
        components.append({
            "id":           parent_id,
            "question":     question_id,
            "part":         part_id,
            "topic":        topic_name,
            "text":         enriched_text,
            "raw_text":     raw_text,
            "content_only": content_only,
            "weight":       float(main_weight),
            "is_critical":  True
        })

    for i, s in enumerate(subs, 1):
        # If a sub_topic has no heading (e.g. came from a CONTINUATION merge),
        # infer a short label from the first few words of its content
        if not s.get("topic", "").strip() and s.get("content", "").strip():
            s = dict(s)
            s["topic"] = f"Topic {i}"
        flatten_topic(
            s,
            f"{parent_id}.S{i}",
            sub_weight,
            components,
            question_id,
            part_id,
            question_context=question_context,
            main_ratio=main_ratio
        )


def build_weighted_components(extracted_papers, total_marks):
    components    = []
    all_questions = []

    for paper in extracted_papers:
        all_questions.extend(paper.get("questions", []))

    if not all_questions:
        return []

    question_weight = total_marks / len(all_questions)

    for q_index, q in enumerate(all_questions, 1):
        qid   = q.get("question_id", f"Q{q_index}")
        parts = q.get("parts", [])

        if not parts:
            continue

        question_context = " ".join(filter(None, [
            q.get("question_title", ""),
            str(qid)
        ])).strip()

        part_weight = question_weight / len(parts)

        for p_index, part in enumerate(parts, 1):
            part_id = part.get("part_id", f"P{p_index}")
            flatten_topic(
                part,
                f"{qid}.P{p_index}",
                part_weight,
                components,
                qid,
                part_id,
                question_context=question_context
            )

    return components
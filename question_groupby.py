import re


def normalize_qid(qid):
    """
    Extracts just the number from any question ID format and normalizes it.

    Examples:
      "Answer 02"    → "Q2"
      "Question #02" → "Q2"
      "Q02"          → "Q2"
      "Q2"           → "Q2"
      "2"            → "Q2"
    """
    numbers = re.findall(r'\d+', str(qid))
    if numbers:
        return f"Q{int(numbers[0])}"  # int() strips leading zeros
    return str(qid).strip().upper()


def group_by_question(components):
    """
    Groups components by normalized question ID.
    Ensures model "Answer 02" and student "Question #02" both map to "Q2".
    """
    grouped = {}
    for c in components:
        normalized = normalize_qid(c["question"])
        if normalized not in grouped:
            grouped[normalized] = []
        grouped[normalized].append(c)
    return grouped
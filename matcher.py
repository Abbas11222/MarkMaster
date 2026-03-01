import warnings
import logging
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =====================================================
# LOAD MODELS ONCE
# BERTScorer      → token-level, best for similar structure
# SentenceTransformer → sentence-level, best for different structure
# =====================================================

print("[matcher] Loading BERTScorer (one-time)...")
_scorer = BERTScorer(
    model_type="distilbert-base-uncased",   # 40% faster than roberta-large, still accurate
    lang="en",
    rescale_with_baseline=False,             # baseline not available for distilbert, skip it
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("[matcher] BERTScorer ready.")

print("[matcher] Loading SentenceTransformer (one-time)...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[matcher] SentenceTransformer ready.")


# =====================================================
# BERTSCORE — token level
# Best for: similar structure, same vocabulary
# =====================================================

def bert_score_single(model_text, student_text):
    try:
        model_text   = model_text.strip()
        student_text = student_text.strip()
        if not model_text or not student_text:
            return 0.0
        _, _, F1 = _scorer.score([student_text], [model_text])
        return max(0.0, min(1.0, round(float(F1[0]), 4)))
    except Exception as e:
        print(f"[matcher] BERTScore failed: {e}")
        return 0.0


def bert_score_batch(model_text, student_texts):
    try:
        model_text    = model_text.strip()
        student_texts = [s.strip() for s in student_texts]
        if not model_text or not student_texts:
            return [0.0] * len(student_texts)
        references = [model_text] * len(student_texts)
        _, _, F1   = _scorer.score(student_texts, references)
        return [max(0.0, min(1.0, round(float(f), 4))) for f in F1]
    except Exception as e:
        print(f"[matcher] BERTScore batch failed: {e}")
        return [0.0] * len(student_texts)


# =====================================================
# SENTENCE EMBEDDING SIMILARITY — sentence level
# Best for: different structure, same concept
#
# Key difference from BERTScore:
# BERTScore: "Specification: specify what system does"
#            vs "Goal: prove correctness" → ~0.08 (no token overlap)
#
# Sentence embeddings: both map to "formal methods concepts"
#            region in embedding space → ~0.55-0.70
# =====================================================

def embedding_similarity(text1, text2):
    """
    Cosine similarity between sentence embeddings.
    Captures topic-level semantic similarity regardless of vocabulary.
    Returns 0.0 to 1.0.
    """
    try:
        text1 = text1.strip()
        text2 = text2.strip()
        if not text1 or not text2:
            return 0.0
        emb = _embedder.encode([text1, text2], convert_to_numpy=True)
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return max(0.0, min(1.0, round(float(sim), 4)))
    except Exception as e:
        print(f"[matcher] Embedding similarity failed: {e}")
        return 0.0


def embedding_similarity_batch(model_text, student_texts):
    """Batch version — one model vs many students."""
    try:
        model_text    = model_text.strip()
        student_texts = [s.strip() for s in student_texts]
        if not model_text or not student_texts:
            return [0.0] * len(student_texts)
        all_texts = [model_text] + student_texts
        embeddings = _embedder.encode(all_texts, convert_to_numpy=True)
        model_emb   = embeddings[0:1]
        student_emb = embeddings[1:]
        sims = cosine_similarity(model_emb, student_emb)[0]
        return [max(0.0, min(1.0, round(float(s), 4))) for s in sims]
    except Exception as e:
        print(f"[matcher] Embedding batch failed: {e}")
        return [0.0] * len(student_texts)


        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        vectorizer.fit([model_text])
        keywords = vectorizer.get_feature_names_out()
        if len(keywords) == 0:
            return 0.0
        student_lower = student_text.lower()
        matched = sum(1 for kw in keywords if kw in student_lower)
        return round(matched / len(keywords), 4)
    except Exception:
        return 0.0


# =====================================================
# COMBINED SEMANTIC SCORE
#
# For holistic (full answer vs full answer):
#   70% sentence embedding + 30% BERTScore
#   Embedding dominates because answers are structurally different
#
# For topic-level (topic vs topic):
#   40% sentence embedding + 60% BERTScore
#   BERTScore dominates when comparing specific focused topics
# =====================================================

def holistic_semantic_score(model_text, student_text):
    """
    For comparing full answers that may be structured differently.
    Embedding-dominant: captures same-topic, different-angle answers.
    """
    emb_sim  = embedding_similarity(model_text, student_text)
    bert_sim = bert_score_single(model_text, student_text)
    return round((0.70 * emb_sim) + (0.30 * bert_sim), 4)


def topic_semantic_score(model_text, student_texts):
    """
    For comparing individual topics.
    BERTScore-dominant: precise when matching specific components.
    Falls back to embedding when BERTScore is very low.
    """
    bert_scores = bert_score_batch(model_text, student_texts)
    emb_scores  = embedding_similarity_batch(model_text, student_texts)

    combined = []
    for b, e in zip(bert_scores, emb_scores):
        if b >= 0.45:
            # BERTScore is reliable — use it with small embedding boost
            score = (0.70 * b) + (0.30 * e)
        else:
            # BERTScore low — structurally different answer
            # Give more weight to embedding which handles this better
            score = (0.40 * b) + (0.60 * e)
        combined.append(round(score, 4))

    return combined


# =====================================================
# SCORE CURVE
# =====================================================

def apply_score_curve(score):
    """
    Pure linear mapping. Works for both:
    - Sentence embedding similarity (0-1 range)
    - BERTScore without baseline rescaling (0-1 range)

    Below 0.20 = off-topic = 0 marks.
    """
    if score < 0.20:
        return 0.0
    return round(score, 4)


def grade_label(score_ratio):
    if score_ratio >= 0.90:
        return "Excellent"
    elif score_ratio >= 0.70:
        return "Good"
    elif score_ratio >= 0.45:
        return "Partial"
    elif score_ratio > 0:
        return "Weak"
    else:
        return "Not Attempted"


# =====================================================
# MERGE TEXTS
# =====================================================

def merge_content_only(components):
    """Pure answer content, no headings, no prefixes."""
    return " ".join(
        c.get("content_only", c.get("raw_text", c.get("text", ""))).strip()
        for c in components
        if c.get("content_only", c.get("raw_text", c.get("text", ""))).strip()
    )


# =====================================================
# MAIN SCORING
# =====================================================

def score_student_answer(
    model_components,
    student_components,
    total_marks,
    threshold=0.25
):
    """
    Three-layer scoring — no extra API calls:

    1. Holistic (50%): full model answer vs full student answer
       Uses embedding-dominant combined score
       Handles structurally different but conceptually correct answers

    2. Topic-level (50%): each model topic vs best matching student topic
       Uses adaptive BERTScore+embedding blend
       Falls back to embedding when BERTScore is low

    3. Coverage: keyword presence check
    """
    if not model_components:
        return 0.0, []

    breakdown = []

    if not student_components:
        for m in model_components:
            if not m.get("topic", "").strip():
                continue
            breakdown.append({
                "topic":                    m["topic"],
                "marks_available":          float(round(m["weight"], 2)),
                "marks_earned":             0.0,
                "contextual_understanding": 0.0,
                "grade":                    "Not Attempted"
            })
        return 0.0, breakdown

    valid_model = [
        m for m in model_components
        if m.get("text", "").strip() and m.get("topic", "").strip()
    ]

    if not valid_model:
        return 0.0, []

    valid_total_weight = sum(m["weight"] for m in valid_model)
    student_texts      = [c.get("text", "").strip() for c in student_components]

    # ── Holistic: embedding-dominant ──────────────────────────────
    model_content   = merge_content_only(valid_model)
    student_content = merge_content_only(student_components)

    holistic_sim   = holistic_semantic_score(model_content, student_content)
    holistic_ratio = apply_score_curve(holistic_sim)
    holistic_score = holistic_ratio * total_marks

    breakdown.append({
        "topic":                    "Overall Answer",
        "marks_available":          float(round(total_marks, 2)),
        "marks_earned":             float(round(holistic_score, 2)),
        "contextual_understanding": float(round(holistic_sim, 4)),
        "grade":                    grade_label(holistic_ratio)
    })

    # ── Topic-level: adaptive BERTScore+embedding ─────────────────
    used_students = set()
    topic_total   = 0.0

    for m in valid_model:
        combined_scores = topic_semantic_score(m["text"], student_texts)

        best_j     = -1
        best_score = 0.0
        for j, s in enumerate(combined_scores):
            if j not in used_students and s > best_score:
                best_score = s
                best_j     = j

        marks_earned = 0.0
        ratio        = 0.0

        if best_score >= threshold and best_j != -1:
            ratio        = apply_score_curve(best_score)
            weight_prop  = m["weight"] / valid_total_weight
            marks_earned = ratio * weight_prop * total_marks
            used_students.add(best_j)

        topic_total += marks_earned

        breakdown.append({
            "topic":                    m["topic"],
            "marks_available":          float(round(
                (m["weight"] / valid_total_weight) * total_marks, 2
            )),
            "marks_earned":             float(round(marks_earned, 2)),
            "contextual_understanding": float(round(best_score, 4)),
            "grade":                    grade_label(ratio)
        })

    # ── Final score: best of holistic or topic ───────────────────
    # Always give student the higher of the two scores.
    # - Topic sum: precise per-point matching
    # - Holistic:  catches correct answers with different structure
    # Student always gets benefit of the doubt.
    final_score = max(topic_total, holistic_score)
    final_score = max(0.0, min(float(total_marks), float(final_score)))

    return float(round(final_score, 2)), breakdown
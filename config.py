"""
config.py
─────────
Single source of truth for all shared models and API clients.
Every other file imports from here — nothing is initialized twice.

Usage:
    from config import groq_client, LLM_MODEL, embedder, bert_scorer
"""

import os
import warnings
import logging
import torch
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer

load_dotenv()

# ─────────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─────────────────────────────────────────────
# SENTENCE TRANSFORMER
# Used by: matcher.py, vector_store.py
# ─────────────────────────────────────────────
print("[config] Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[config] SentenceTransformer ready.")

# ─────────────────────────────────────────────
# BERT SCORER
# Used by: matcher.py
# ─────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("[config] Loading BERTScorer...")
bert_scorer = BERTScorer(
    model_type="distilbert-base-uncased",
    lang="en",
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("[config] BERTScorer ready.")
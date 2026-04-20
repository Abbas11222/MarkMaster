"""
vector_store.py
───────────────
Stores chunks in ChromaDB (local, no server needed).
Retrieves top-k most relevant chunks for a given query.

DB is stored in ./chroma_db/ folder — persists between runs.
Uses chromadb >= 0.5.x PersistentClient API.
"""

import os
import chromadb
from config import embedder as _embedder_model

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "markmaster_kb"

_chroma_client = None
_collection    = None

CHROMA_AVAILABLE = True  # chromadb is a hard requirement


# ─────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────

def _get_embedder():
    return _embedder_model


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ─────────────────────────────────────────────────────────────────
# STORE
# ─────────────────────────────────────────────────────────────────

def store_chunks(chunks):
    """
    Embed and store a list of chunk dicts in ChromaDB.
    Skips chunks that are already stored (by chunk_id).
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    existing   = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        print("  ℹ️ All chunks already in knowledge base — skipping")
        return 0

    texts     = [c["text"]     for c in new_chunks]
    ids       = [c["chunk_id"] for c in new_chunks]
    metadatas = [{
        "source":      c["source"],
        "page":        c["page"],
        "type":        c["type"],
        "chunk_index": c["chunk_index"]
    } for c in new_chunks]

    print(f"  🔄 Embedding {len(new_chunks)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    batch_size = 500
    for i in range(0, len(new_chunks), batch_size):
        collection.add(
            ids        = ids[i:i+batch_size],
            documents  = texts[i:i+batch_size],
            embeddings = embeddings[i:i+batch_size],
            metadatas  = metadatas[i:i+batch_size]
        )

    print(f"  ✅ Stored {len(new_chunks)} new chunks")
    return len(new_chunks)


def clear_knowledge_base():
    """Delete all stored chunks — full reset."""
    global _collection, _chroma_client
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(COLLECTION_NAME)
        _collection    = None
        _chroma_client = None
        print("  🗑️ Knowledge base cleared")
    except Exception as e:
        print(f"  ⚠️ Could not clear knowledge base: {e}")


def get_kb_stats():
    """Return how many chunks are stored."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────────────────────────

def retrieve(query, top_k=6):
    """
    Given a question string, return the top_k most relevant chunks.
    Returns list of { text, source, page, score }
    """
    collection = _get_collection()
    embedder   = _get_embedder()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":   doc,
            "source": meta.get("source", ""),
            "page":   meta.get("page", 0),
            "score":  round(1 - dist, 4)
        })

    return chunks
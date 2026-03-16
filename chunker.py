"""
chunker.py
──────────
Splits loaded pages into overlapping text chunks for vector storage.

chunk_size    = 400 words  (enough context per chunk)
chunk_overlap = 60 words   (overlap prevents boundary information loss)

Each chunk = {
    "chunk_id": str,
    "text":     str,
    "source":   str,
    "page":     int,
    "type":     str,
    "chunk_index": int
}
"""


def _split_words(text, size, overlap):
    """
    Split text into overlapping word-based chunks.
    Returns list of chunk strings.
    """
    words = text.split()
    if len(words) <= size:
        return [text]

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap   # step forward by (size - overlap)
    return chunks


def chunk_pages(pages, chunk_size=400, chunk_overlap=60):
    """
    Takes list of page dicts from loader.py
    Returns list of chunk dicts ready for embedding.
    """
    all_chunks = []
    chunk_counter = 0

    for page in pages:
        text   = page.get("text", "").strip()
        source = page.get("source", "unknown")
        pg     = page.get("page", 1)
        ptype  = page.get("type", "unknown")

        if not text:
            continue

        splits = _split_words(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(splits):
            if len(chunk_text.strip()) < 20:   # skip tiny fragments
                continue

            all_chunks.append({
                "chunk_id":    f"{source}_p{pg}_c{i}",
                "text":        chunk_text.strip(),
                "source":      source,
                "page":        pg,
                "type":        ptype,
                "chunk_index": chunk_counter
            })
            chunk_counter += 1

    print(f"  ✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

import re
from collections import Counter


def chunk_text(text, chunk_size=1800, overlap=180):
    """Fast local chunking; no external API calls."""
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Prefer a paragraph/sentence boundary near the end of the window.
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + int(chunk_size * 0.65):
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def create_embeddings(chunks):
    """Compatibility stub. Retrieval is local and does not require paid embeddings."""
    return []


def _tokens(value):
    return re.findall(r"[\w\u0600-\u06FF]+", (value or "").lower())


def retrieve_chunks(chunks, embeddings=None, query="", top_k=5):
    """Fast request-local lexical ranking; safe across serverless instances."""
    if not chunks:
        return []
    query_terms = set(_tokens(query))
    if not query_terms:
        selected = chunks[:top_k]
        return [{"chunk": item, "score": 0.0} for item in selected]
    scored = []
    for index, chunk in enumerate(chunks):
        terms = _tokens(chunk)
        if not terms:
            continue
        counts = Counter(terms)
        overlap = sum(counts[t] for t in query_terms)
        coverage = len(query_terms.intersection(counts)) / max(1, len(query_terms))
        density = overlap / max(1, len(terms))
        score = coverage + density
        scored.append((score, index, chunk))
    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    selected = scored[:max(1, top_k)]
    # If query words do not occur (e.g. semantic paraphrase), preserve source order.
    if not selected or selected[0][0] == 0:
        selected = [(0.0, i, chunk) for i, chunk in enumerate(chunks[:max(1, top_k)])]
    return [{"chunk": chunk, "score": score} for score, _, chunk in selected]


def build_context(retrieved_chunks):
    if not retrieved_chunks:
        return ""
    return "\n\n---\n\n".join(item["chunk"] for item in retrieved_chunks if item.get("chunk"))

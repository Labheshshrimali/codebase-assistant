"""Rerank candidates using a cross-encoder to improve accuracy.
Using ms-marco-MiniLM as it fits well in memory for local inference.
"""

from sentence_transformers import CrossEncoder

from app.ingestion.chunker import Chunk

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # swap for a code-tuned reranker later


def rerank(query: str, chunks: list[Chunk], top_n: int = 6) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []
    pairs = [(query, c.code) for c in chunks]
    scores = _reranker.predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_n]

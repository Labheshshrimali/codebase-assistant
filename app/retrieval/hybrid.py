"""Fuse vector, call-graph, and BM25 candidates, then rerank to a final context set.
Provides the main entrypoint for the retrieval pipeline.
"""

from dataclasses import dataclass

from app.ingestion.chunker import Chunk
from app.retrieval.indexer import RepoIndex
from app.retrieval.rerank import rerank


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    sources: list[str]  # e.g. ["vector", "bm25"]


def retrieve(
    index: RepoIndex,
    query: str,
    top_k_candidates: int = 20,
    top_n_final: int = 6,
    use_graph: bool = True,
    use_bm25: bool = True,
) -> list[RetrievalResult]:
    candidate_sources: dict[int, set[str]] = {}

    for idx in index.vector_search(query, top_k=top_k_candidates):
        candidate_sources.setdefault(idx, set()).add("vector")

    if use_bm25:
        for idx in index.bm25_search(query, top_k=top_k_candidates):
            candidate_sources.setdefault(idx, set()).add("bm25")

    if use_graph:
        vector_hits = [i for i, s in candidate_sources.items() if "vector" in s]
        for idx in index.graph_neighbors(vector_hits, depth=1):
            candidate_sources.setdefault(idx, set()).add("graph")

    candidate_idxs = list(candidate_sources.keys())
    candidate_chunks = [index.chunks[i] for i in candidate_idxs]

    reranked = rerank(query, candidate_chunks, top_n=top_n_final)

    results = []
    for chunk, score in reranked:
        orig_idx = next(i for i in candidate_idxs if index.chunks[i].id == chunk.id)
        results.append(RetrievalResult(
            chunk=chunk,
            score=score,
            sources=sorted(candidate_sources[orig_idx]),
        ))
    return results

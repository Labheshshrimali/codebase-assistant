"""Evaluation harness to measure retrieval performance.

Usage:
    python -m app.eval.run_eval --repo psf__requests --dataset app/eval/qa_dataset.jsonl
"""

import argparse
import json
from pathlib import Path

from app.retrieval.hybrid import retrieve
from app.retrieval.indexer import RepoIndex


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_chunk_id(chunk_id: str) -> tuple[str, int, int]:
    """'src/requests/adapters.py:201-222' -> ('src/requests/adapters.py', 201, 222)"""
    path, line_range = chunk_id.rsplit(":", 1)
    start, end = line_range.split("-")
    return path, int(start), int(end)


def _overlaps(retrieved_id: str, relevant_id: str) -> bool:
    """A retrieved chunk counts as a match if it's in the same file and its
    line range overlaps the ground-truth range at all — not an exact string
    match. Hand-counted ground truth line numbers rarely line up exactly
    with AST-parsed chunk boundaries (off-by-one from blank lines, decorators,
    etc.), so exact-match precision/recall would understate real performance.
    """
    try:
        r_path, r_start, r_end = _parse_chunk_id(retrieved_id)
        g_path, g_start, g_end = _parse_chunk_id(relevant_id)
    except ValueError:
        return retrieved_id == relevant_id
    return r_path == g_path and r_start <= g_end and g_start <= r_end


def precision_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> tuple[float, float]:
    if not retrieved_ids:
        return 0.0, 0.0
    hits_retrieved = sum(
        1 for r in retrieved_ids if any(_overlaps(r, g) for g in relevant_ids)
    )
    hits_relevant = sum(
        1 for g in relevant_ids if any(_overlaps(r, g) for r in retrieved_ids)
    )
    precision = hits_retrieved / len(retrieved_ids)
    recall = hits_relevant / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def evaluate_naive_vector(index: RepoIndex, dataset: list[dict], top_n: int = 6) -> dict:
    """Simulates a typical tutorial-style RAG baseline: raw vector similarity
    top-k, no reranking, no BM25/graph fusion. This is what most basic RAG
    projects actually ship — a fairer comparison point than a reranked
    vector-only pipeline, which already benefits from the reranker.
    """
    precisions, recalls = [], []
    for item in dataset:
        top_idxs = index.vector_search(item["question"], top_k=top_n)
        retrieved_ids = [index.chunks[i].id for i in top_idxs]
        p, r = precision_recall_at_k(retrieved_ids, item["relevant_chunk_ids"])
        precisions.append(p)
        recalls.append(r)
    n = len(dataset) or 1
    return {
        "precision_at_k": sum(precisions) / n,
        "recall_at_k": sum(recalls) / n,
        "n_questions": len(dataset),
    }


def evaluate(index: RepoIndex, dataset: list[dict], use_graph: bool, use_bm25: bool) -> dict:
    precisions, recalls = [], []
    for item in dataset:
        results = retrieve(index, item["question"], use_graph=use_graph, use_bm25=use_bm25)
        retrieved_ids = [r.chunk.id for r in results]
        p, r = precision_recall_at_k(retrieved_ids, item["relevant_chunk_ids"])
        precisions.append(p)
        recalls.append(r)
    n = len(dataset) or 1
    return {
        "precision_at_k": sum(precisions) / n,
        "recall_at_k": sum(recalls) / n,
        "n_questions": len(dataset),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="repo slug, e.g. psf__requests")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    index = RepoIndex(repo_slug=args.repo, qdrant_url=args.qdrant_url)
    # NOTE: assumes /ingest has already been run for this repo so the
    # Qdrant collection + BM25 index exist. Wire up index loading here
    # once you add persistence (currently index.build() is in-memory-ish
    # for the graph/BM25 parts — see indexer.py roadmap note).

    baseline = evaluate(index, dataset, use_graph=False, use_bm25=False)
    hybrid = evaluate(index, dataset, use_graph=True, use_bm25=True)

    print("=== Baseline (vector-only) ===")
    print(json.dumps(baseline, indent=2))
    print("\n=== Hybrid (vector + graph + bm25) ===")
    print(json.dumps(hybrid, indent=2))

    p_gain = (hybrid["precision_at_k"] - baseline["precision_at_k"]) / max(baseline["precision_at_k"], 1e-6) * 100
    r_gain = (hybrid["recall_at_k"] - baseline["recall_at_k"]) / max(baseline["recall_at_k"], 1e-6) * 100
    print(f"\nPrecision improvement: {p_gain:+.1f}%")
    print(f"Recall improvement:    {r_gain:+.1f}%")


if __name__ == "__main__":
    main()

"""Ingest a repo and immediately run eval against it in the same process.

This works around the current in-memory index limitation: the FastAPI
server keeps indexes in memory per-process, so a standalone `run_eval.py`
can't see indexes built by a separate `uvicorn` process. This script
builds the index and evaluates it in one process instead.

Usage:
    python -m app.eval.ingest_and_eval --repo-url https://github.com/psf/requests --dataset app/eval/qa_dataset.jsonl
"""

import argparse
import json

from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo, list_source_files, repo_slug
from app.retrieval.indexer import RepoIndex
from app.eval.run_eval import evaluate, evaluate_naive_vector, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    print(f"Cloning and chunking {args.repo_url}...")
    repo_path = clone_repo(args.repo_url)
    slug = repo_slug(args.repo_url)
    files = list_source_files(repo_path)
    chunks = chunk_repo(repo_path, files)
    print(f"  {len(files)} files, {len(chunks)} chunks")

    print("Building indexes (vector + graph + BM25)...")
    index = RepoIndex(repo_slug=slug)
    index.build(chunks)

    dataset = load_dataset(args.dataset)
    print(f"\nRunning eval on {len(dataset)} hand-labeled questions...\n")

    naive = evaluate_naive_vector(index, dataset)
    baseline = evaluate(index, dataset, use_graph=False, use_bm25=False)
    hybrid = evaluate(index, dataset, use_graph=True, use_bm25=True)

    print("=== Naive (vector top-k, no reranking — typical tutorial RAG) ===")
    print(json.dumps(naive, indent=2))
    print("\n=== Reranked (vector + reranker, no fusion) ===")
    print(json.dumps(baseline, indent=2))
    print("\n=== Hybrid (vector + graph + bm25 + reranker) ===")
    print(json.dumps(hybrid, indent=2))

    def pct_gain(new, old):
        return (new - old) / max(old, 1e-6) * 100

    print(f"\nReranking alone vs naive: precision {pct_gain(baseline['precision_at_k'], naive['precision_at_k']):+.1f}%, recall {pct_gain(baseline['recall_at_k'], naive['recall_at_k']):+.1f}%")
    print(f"Full hybrid vs naive:     precision {pct_gain(hybrid['precision_at_k'], naive['precision_at_k']):+.1f}%, recall {pct_gain(hybrid['recall_at_k'], naive['recall_at_k']):+.1f}%")
    print(f"Full hybrid vs reranked:  precision {pct_gain(hybrid['precision_at_k'], baseline['precision_at_k']):+.1f}%, recall {pct_gain(hybrid['recall_at_k'], baseline['recall_at_k']):+.1f}%")


if __name__ == "__main__":
    main()

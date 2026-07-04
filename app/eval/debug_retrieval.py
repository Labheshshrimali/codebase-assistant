"""Diagnostic: prints the actual retrieved chunk IDs for naive vs hybrid
retrieval on one question, so we can see whether they're genuinely
identical (a real finding) or suspiciously identical (a bug).

Usage:
    python -m app.eval.debug_retrieval --repo-url https://github.com/psf/requests --question "Where is the retry logic implemented?"
"""

import argparse

from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo, list_source_files, repo_slug
from app.retrieval.indexer import RepoIndex
from app.retrieval.hybrid import retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    repo_path = clone_repo(args.repo_url)
    slug = repo_slug(args.repo_url)
    files = list_source_files(repo_path)
    chunks = chunk_repo(repo_path, files)

    index = RepoIndex(repo_slug=slug)
    index.build(chunks)

    print(f"Question: {args.question}\n")

    naive_idxs = index.vector_search(args.question, top_k=6)
    naive_ids = [index.chunks[i].id for i in naive_idxs]
    print("NAIVE (raw vector top-6, no rerank):")
    for cid in naive_ids:
        print(f"  {cid}")

    reranked = retrieve(index, args.question, use_graph=False, use_bm25=False)
    print("\nRERANKED (vector candidates -> reranker):")
    for r in reranked:
        print(f"  {r.chunk.id}   score={r.score:.4f}   sources={r.sources}")

    hybrid = retrieve(index, args.question, use_graph=True, use_bm25=True)
    print("\nHYBRID (vector+graph+bm25 candidates -> reranker):")
    for r in hybrid:
        print(f"  {r.chunk.id}   score={r.score:.4f}   sources={r.sources}")

    print(f"\nNaive == Reranked chunk sets: {set(naive_ids) == {r.chunk.id for r in reranked}}")
    print(f"Reranked == Hybrid chunk sets: {set(r.chunk.id for r in reranked) == set(r.chunk.id for r in hybrid)}")


if __name__ == "__main__":
    main()

"""Finds candidate chunks matching a keyword, using the SAME chunking logic
as the real pipeline — guarantees the chunk_id you get is exactly what
retrieval will produce, no manual line-counting, no off-by-one errors.

Usage:
    python -m app.eval.find_chunks --repo-url https://github.com/psf/requests --keyword cookiejar
"""

import argparse

from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo, list_source_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--max-results", type=int, default=8)
    args = parser.parse_args()

    repo_path = clone_repo(args.repo_url)
    files = list_source_files(repo_path)
    chunks = chunk_repo(repo_path, files)

    keyword = args.keyword.lower()
    matches = [c for c in chunks if keyword in c.code.lower() or keyword in c.name.lower()]

    print(f"Found {len(matches)} chunks matching '{args.keyword}' (showing up to {args.max_results}):\n")
    for c in matches[: args.max_results]:
        print(f'  "{c.id}"   <- {c.node_type} {c.name}')


if __name__ == "__main__":
    main()

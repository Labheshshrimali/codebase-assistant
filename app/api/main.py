from fastapi import FastAPI
from pydantic import BaseModel

from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo, list_source_files, repo_slug
from app.retrieval.hybrid import retrieve
from app.retrieval.indexer import RepoIndex
from app.api.generate import answer_question

app = FastAPI(title="Codebase Assistant")

_indexes: dict[str, RepoIndex] = {}


class IngestRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    repo: str
    question: str


def _get_or_load_index(slug: str) -> RepoIndex | None:
    if slug in _indexes:
        return _indexes[slug]
    index = RepoIndex(repo_slug=slug)
    if index.is_cached() and index.load():
        _indexes[slug] = index
        return index
    return None


@app.post("/ingest")
def ingest(req: IngestRequest):
    slug = repo_slug(req.repo_url)

    cached = _get_or_load_index(slug)
    if cached is not None:
        return {
            "repo": slug,
            "files_indexed": len({c.repo_relative_path for c in cached.chunks}),
            "chunks_indexed": len(cached.chunks),
            "cached": True,
        }

    repo_path = clone_repo(req.repo_url)
    files = list_source_files(repo_path)
    chunks = chunk_repo(repo_path, files)

    index = RepoIndex(repo_slug=slug)
    index.build(chunks)
    _indexes[slug] = index

    return {
        "repo": slug,
        "files_indexed": len(files),
        "chunks_indexed": len(chunks),
        "cached": False,
    }


@app.post("/ask")
def ask(req: AskRequest):
    index = _get_or_load_index(req.repo)
    if index is None:
        return {"error": f"repo '{req.repo}' not indexed yet. POST /ingest first."}
    results = retrieve(index, req.question)
    return answer_question(req.question, results)


@app.get("/health")
def health():
    return {"status": "ok", "indexed_repos": list(_indexes.keys())}
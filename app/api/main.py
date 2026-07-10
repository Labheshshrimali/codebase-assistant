from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional

from app.core.config import setup_logging, validate_config, logger
from app.core.exceptions import IngestionError
from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo, list_source_files, repo_slug
from app.retrieval.hybrid import retrieve
from app.retrieval.indexer import RepoIndex
from app.api.generate import answer_question

# Initialize professional logging
setup_logging()

# Validate environment variables before starting the app
try:
    validate_config()
except RuntimeError as e:
    logger.error(str(e))
    raise e

app = FastAPI(
    title="Codebase Assistant API",
    description="Hybrid RAG API for querying GitHub repositories",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_indexes: Dict[str, RepoIndex] = {}


class IngestRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    repo: str
    question: str


def _get_or_load_index(slug: str) -> Optional[RepoIndex]:
    if slug in _indexes:
        return _indexes[slug]
    index = RepoIndex(repo_slug=slug)
    if index.is_cached() and index.load():
        _indexes[slug] = index
        return index
    return None


@app.post(
    "/ingest", 
    tags=["Indexing"], 
    summary="Index a GitHub Repository", 
    description="Clones the repository, parses code into AST chunks, and builds a hybrid index (Vector + BM25 + Graph)."
)
def ingest(req: IngestRequest):
    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only https://github.com/ URLs are supported.")

    try:
        slug = repo_slug(req.repo_url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid repository URL format.")

    cached = _get_or_load_index(slug)
    if cached is not None:
        return {
            "repo": slug,
            "files_indexed": len({c.repo_relative_path for c in cached.chunks}),
            "chunks_indexed": len(cached.chunks),
            "cached": True,
        }

    try:
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
    except Exception as e:
        logger.error(f"Ingestion failed for {req.repo_url}: {str(e)}")
        raise IngestionError(f"Failed to ingest repository: {str(e)}")


@app.post(
    "/ask", 
    tags=["Query"], 
    summary="Ask a question about the code", 
    description="Retrieves relevant code chunks using hybrid search and generates an answer using an LLM."
)
def ask(req: AskRequest):
    index = _get_or_load_index(req.repo)
    if index is None:
        return {"error": f"repo '{req.repo}' not indexed yet. POST /ingest first."}

    try:
        results = retrieve(index, req.question)
        return answer_question(req.question, results)
    except Exception as e:
        logger.error(f"Query failed for repo {req.repo}: {str(e)}")
        return {"error": f"An internal error occurred during retrieval: {str(e)}"}


@app.get(
    "/health", 
    tags=["System"], 
    summary="Check system health"
)
def health():
    return {"status": "ok", "indexed_repos": list(_indexes.keys())}


# Mount the frontend directory to serve the UI at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
"""Clone a GitHub repo locally and list source files to ingest."""

from pathlib import Path
from git import Repo

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SUPPORTED_EXTENSIONS = {".py"}  # extend as you add tree-sitter grammars


def repo_slug(repo_url: str) -> str:
    """https://github.com/psf/requests -> psf__requests"""
    parts = repo_url.rstrip("/").split("/")[-2:]
    return "__".join(parts).replace(".git", "")


def clone_repo(repo_url: str) -> Path:
    slug = repo_slug(repo_url)
    dest = DATA_DIR / "repos" / slug
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(repo_url, dest, depth=1)
    return dest


def list_source_files(repo_path: Path) -> list[Path]:
    ignore_dirs = {".git", "node_modules", "venv", "__pycache__", ".mypy_cache"}
    files = []
    for path in repo_path.rglob("*"):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/psf/requests"
    path = clone_repo(url)
    files = list_source_files(path)
    print(f"Cloned to {path}, found {len(files)} source files")

"""Build the three indexes the hybrid retriever needs:
  1. vector index (Qdrant)      -> semantic similarity
  2. call/import graph (networkx) -> "what calls / is called by this"
  3. BM25 keyword index           -> exact-term recall (error codes, var names)
"""

import re
from pathlib import Path

import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # swap for a stronger code model, e.g. jina-embeddings-v2-base-code
VECTOR_SIZE = 384

_embedder = SentenceTransformer(EMBED_MODEL_NAME)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


import os

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
_call_parser = Parser(PY_LANGUAGE)


def _extract_called_names(code: str) -> set[str]:
    """Parse code and return the set of function/method names actually
    CALLED (real 'call' AST nodes), not just any text occurrence of a name.
    This avoids false positives from comments, strings, or unrelated
    identifiers that the old regex approach would incorrectly match.
    """
    tree = _call_parser.parse(code.encode("utf-8"))
    called_names: set[str] = set()

    def walk(node):
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                if func_node.type == "identifier":
                    called_names.add(code[func_node.start_byte:func_node.end_byte])
                elif func_node.type == "attribute":
                    attr_node = func_node.child_by_field_name("attribute")
                    if attr_node is not None:
                        called_names.add(code[attr_node.start_byte:attr_node.end_byte])
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return called_names


class RepoIndex:
    def __init__(self, repo_slug: str, qdrant_url: str = None):
        self.repo_slug = repo_slug
        self.collection = f"repo_{repo_slug}"
        resolved_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=resolved_url)
        self.graph = nx.DiGraph()
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        # 1. vector index
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        embeddings = _embedder.encode([c.code for c in chunks], show_progress_bar=True)
        points = [
            PointStruct(id=i, vector=emb.tolist(), payload={"chunk_id": c.id})
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upload_points(collection_name=self.collection, points=points)

        # 2. call/import graph — real AST-based call detection: parses each
        # chunk and looks for actual 'call' nodes, not just any text
        # occurrence of a name (which the old regex approach incorrectly
        # matched inside comments, strings, or unrelated identifiers).
        names_to_chunks: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            names_to_chunks.setdefault(c.name, []).append(i)

        for i, c in enumerate(chunks):
            self.graph.add_node(i, chunk_id=c.id)
            try:
                called_names = _extract_called_names(c.code)
            except Exception:
                called_names = set()  # malformed snippet — skip edges for it
            for name in called_names:
                if name == c.name:
                    continue  # skip self-recursion edges
                for j in names_to_chunks.get(name, []):
                    self.graph.add_edge(i, j)  # i genuinely calls j

        # 3. BM25 keyword index
        tokenized = [_tokenize(c.code) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, top_k: int = 10) -> list[int]:
        query_vec = _embedder.encode(query).tolist()
        hits = self.client.search(collection_name=self.collection, query_vector=query_vec, limit=top_k)
        return [h.id for h in hits]

    def bm25_search(self, query: str, top_k: int = 10) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def graph_neighbors(self, chunk_idxs: list[int], depth: int = 1) -> list[int]:
        neighbors: set[int] = set()
        frontier = set(chunk_idxs)
        for _ in range(depth):
            next_frontier = set()
            for i in frontier:
                next_frontier |= set(self.graph.successors(i)) | set(self.graph.predecessors(i))
            neighbors |= next_frontier
            frontier = next_frontier
        return list(neighbors - set(chunk_idxs))

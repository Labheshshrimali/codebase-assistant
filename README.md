# Intelligent Codebase Assistant

Ask deep questions about any GitHub repo and get accurate, code-aware answers
with citations to exact file/line locations. Built with AST-aware chunking,
hybrid retrieval (vector + graph + keyword), reranking, and an evaluation
harness — not a wrapper around a single API call.

## Why this exists

Most "chat with your codebase" demos are naive vector search over raw text
chunks. This project is built to actually work on large, real repos:

- **AST-based chunking** — chunks are functions/classes, not arbitrary line
  windows, so retrieved context is always syntactically complete.
- **Hybrid retrieval** — vector similarity + a lightweight call/import graph +
  BM25 keyword fallback, combined and reranked.
- **Evaluation harness** — retrieval precision/recall and answer correctness
  measured against a hand-labeled QA set, not just eyeballed.
- **Deployable** — Dockerized FastAPI service with streaming responses.

## Architecture

```
GitHub URL
   │
   ▼
[ingestion] clone repo → parse with tree-sitter → chunk by function/class
   │
   ▼
[indexing] embed chunks (sentence-transformers) → Qdrant (vector)
           build call/import graph → networkx (graph)
           build keyword index → rank-bm25
   │
   ▼
[retrieval] vector top-k  +  graph neighbors  +  BM25 top-k
                    │
                    ▼
              [rerank] cross-encoder reranker → top-N final context
                    │
                    ▼
[generation] LLM synthesizes answer with file:line citations
                    │
                    ▼
[api] FastAPI, streaming SSE endpoint
```

## Project layout

```
app/
  ingestion/
    clone.py          # clone + walk repo
    chunker.py         # tree-sitter based AST chunking
  retrieval/
    indexer.py         # build vector + graph + keyword indexes
    hybrid.py           # combine vector/graph/BM25 retrieval
    rerank.py            # cross-encoder reranking
  api/
    main.py               # FastAPI app, /ingest and /ask endpoints
    generate.py            # LLM call + citation formatting
  eval/
    qa_dataset.jsonl        # hand-labeled question/answer pairs per repo
    run_eval.py               # computes retrieval + answer metrics
data/                          # cloned repos + built indexes (gitignored)
tests/
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker compose up -d qdrant   # starts vector DB locally
```

## Run

```bash
uvicorn app.api.main:app --reload
```

Then:
```bash
curl -X POST localhost:8000/ingest -d '{"repo_url": "https://github.com/psf/requests"}'
curl -X POST localhost:8000/ask -d '{"repo": "requests", "question": "where is retry logic handled?"}'
```

## Evaluation

```bash
python -m app.eval.ingest_and_eval --repo-url https://github.com/psf/requests --dataset app/eval/qa_dataset.jsonl
```

Reports retrieval precision@k / recall@k against baseline (pure vector
search) vs. the hybrid pipeline, plus answer-correctness scoring.

### Results (on psf/requests, N=10 hand-labeled questions)

<!-- Fill in after running the eval above -->
| Method | Precision@k | Recall@k |
|---|---|---|
| Baseline (vector-only) | — | — |
| Hybrid (vector + graph + BM25) | — | — |

**Finding:** _[fill in — e.g. "On this small, well-documented repo, hybrid
retrieval showed [X]% improvement, concentrated in exact-term and
relationship-based queries (e.g. 'what calls X'), where pure vector search
structurally cannot help. On straightforward semantic queries, vector-only
search already performed well, suggesting hybrid retrieval's advantage
grows with codebase size/complexity rather than being uniform."]_


## Roadmap / what to build next

- [ ] Multi-language tree-sitter grammars (currently Python-first)
- [ ] Incremental re-indexing on repo updates instead of full rebuild
- [ ] Fine-tune the reranker on the code-QA eval set once it's large enough
- [ ] Load test with Locust, document latency at scale
- [ ] Frontend chat UI with inline code citation rendering

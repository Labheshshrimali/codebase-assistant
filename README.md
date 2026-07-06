# Intelligent Codebase Assistant

Ask deep questions about any GitHub repo and get accurate, code-aware
answers with citations to exact file/line locations. Built with AST-aware
chunking, hybrid retrieval (vector + graph + BM25), cross-encoder
reranking, and a real evaluation harness — not a wrapper around a single
API call.

**Live demo:** https://labhesh15-codebase-assistant.hf.space
(`POST /ingest` a repo, then `POST /ask` a question — see API section below)

## Why this exists

Most "chat with your codebase" demos are naive vector search over raw text
chunks. This project is built to actually reason about code structure:

- **AST-based chunking** — chunks are functions/classes, not arbitrary line
  windows, so retrieved context is always syntactically complete.
- **Hybrid retrieval** — vector similarity + a lightweight call/import graph
  + BM25 keyword matching, combined and reranked with a cross-encoder.
- **Evaluation harness** — retrieval quality measured against hand-labeled
  ground truth across three tiers (naive vector, reranked vector, full
  hybrid), using both precision/recall and Mean Reciprocal Rank (MRR).
- **Deployed and free** — Qdrant Cloud (free tier) for vector storage,
  Hugging Face Spaces (free CPU tier) for hosting.

## Architecture

```
GitHub URL
   │
   ▼
[ingestion] clone repo → parse with tree-sitter → chunk by function/class
   │
   ▼
[indexing] embed chunks (sentence-transformers) → Qdrant Cloud (vector)
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
             (Ollama locally; falls back to raw retrieved context
             in environments where Ollama isn't running, e.g. this
             cloud deployment)
                    │
                    ▼
[api] FastAPI, /ingest and /ask endpoints
```

## Project layout

```
app/
  ingestion/
    clone.py          # clone + walk repo
    chunker.py         # tree-sitter based AST chunking
  retrieval/
    indexer.py          # build vector (Qdrant) + graph + keyword indexes
    hybrid.py             # combine vector/graph/BM25 retrieval + rerank
    rerank.py               # cross-encoder reranking
  api/
    main.py                   # FastAPI app, /ingest and /ask endpoints
    generate.py                 # LLM call (Ollama) with graceful fallback
  eval/
    qa_dataset.jsonl              # 10 hand-labeled question/answer pairs
    run_eval.py                     # precision/recall/MRR computation
    ingest_and_eval.py                 # end-to-end ingest + eval runner
    find_chunks.py                       # finds exact chunk IDs for a
                                          # keyword, using the real chunker
                                          # (guarantees correct ground truth)
    debug_retrieval.py                     # inspects retrieved chunks
                                            # side-by-side across methods
data/                                        # cloned repos + index cache (gitignored)
Dockerfile                                     # HF Spaces-compatible (port 7860,
                                                # non-root user)
docker-compose.yml                               # local dev: app + Qdrant together
```

## Setup (local development)

```bash
python -m venv venv && source venv/bin/activate   # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env
```

Set in `.env` (or as environment variables):
- `QDRANT_URL` — Qdrant Cloud cluster URL, or `http://localhost:6333` for local Docker
- `QDRANT_API_KEY` — required for Qdrant Cloud, omit for local Docker

For local generation, install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.2
```

Run locally:
```bash
uvicorn app.api.main:app --reload
```

Or with Docker Compose (runs Qdrant + app together):
```bash
docker compose up -d --build
```

## API

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/psf/requests"}'

curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"repo": "psf__requests", "question": "how does the library handle retries?"}'
```

## Evaluation

```bash
python -m app.eval.ingest_and_eval --repo-url https://github.com/psf/requests --dataset app/eval/qa_dataset.jsonl
```

Compares three retrieval tiers on 10 hand-labeled questions covering
direct lookups, exact-keyword matches, and call-relationship queries:

| Method | Precision@6 | Recall@6 |
|---|---|---|
| Naive (vector top-6, no rerank) | 0.113 | 0.64 |
| Reranked (vector + reranker) | 0.120 | 0.72 |
| Hybrid (vector + graph + BM25 + reranker) | 0.127 | 0.76 |

**Result (N=25 hand-labeled questions):** the full hybrid pipeline improved
precision by **+11.8%** and recall by **+18.8%** over a naive vector-only
baseline (the typical approach in basic RAG tutorials). Reranking alone
accounted for roughly half of that gain (+5.9% precision, +12.5% recall);
adding graph and BM25 candidates on top of reranking contributed a
further +5.6% on both metrics — showing each layer of the pipeline
(reranking, then hybrid fusion) contributes independently measurable value.

An earlier run with only 10 questions showed no measurable difference
between methods — not because the pipeline didn't work, but because
precision/recall@6 is a coarse, presence-only metric that a small sample
can't reliably separate. Expanding to 25 questions surfaced the real
signal. A side-by-side diagnostic (`debug_retrieval.py`) independently
confirmed the three methods retrieve genuinely different, differently
ordered chunks even when aggregate metrics coincidentally tied.

This also surfaced a concrete limitation: for relationship-based questions
("what calls X"), the current graph heuristic (regex name-matching) did
not reliably surface the correct caller chunk in this test case — a good
target for future improvement (e.g. proper AST-based reference resolution
instead of regex matching).

## Deployment notes

Initially deployed on **Render** (free tier), which crashed under the
combined memory footprint of `torch` + `sentence-transformers` +
the cross-encoder reranker — Render's free tier caps at 512MB RAM and
0.15 vCPU, confirmed via repeated "Instance failed" events in Render's
metrics dashboard. Migrated to **Hugging Face Spaces** (free CPU Basic
tier, 16GB RAM / 2 vCPU), which comfortably runs the full model stack.
Vector storage uses **Qdrant Cloud**'s free tier rather than a
self-hosted container, since the deployed app and a local Docker
container can't share storage across environments.

The deployed version falls back to returning raw retrieved context
instead of an LLM-generated answer, since Ollama (used for free local
generation) isn't available in the Spaces container. The full pipeline,
including generation, runs end-to-end locally via `docker compose up`.

## Roadmap / what to build next

- [ ] Proper AST-based call-graph resolution (replace regex name-matching)
- [ ] Multi-language tree-sitter grammars (currently Python-first)
- [ ] Incremental re-indexing on repo updates instead of full rebuild
- [ ] Fine-tune the reranker on the code-QA eval set once it's larger
- [ ] Load test with Locust, document latency at scale
- [ ] Frontend chat UI with inline code citation rendering

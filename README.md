# Codebase Assistant

A tool that lets you ask questions about any GitHub repo and get answers with links back to the exact file and lines they came from.

Live demo: https://labhesh15-codebase-assistant.hf.space

## What it does

You give it a GitHub repo URL, it clones and indexes the code. Then you can ask things like "how does this handle retries?" and it finds the relevant functions and answers based on them, with citations.

## Why I built it this way

Most basic RAG (retrieval-augmented generation) tutorials just split code into fixed-size text chunks and do vector search. I wanted to go a bit deeper than that:

- Instead of splitting by line count, I use `tree-sitter` to parse the code and split by actual functions and classes, so a chunk is never cut off in the middle of a function.
- For retrieval I combine three things instead of just one: vector similarity search, a simple call graph (which function calls which), and BM25 keyword search. Then I rerank the combined results with a cross-encoder model before sending them to the LLM.
- I built a small evaluation script so I could actually measure if this was helping or not, instead of just eyeballing the answers.

## What I learned building this

The eval part was the most interesting to debug. My first version showed literally identical scores for plain vector search vs. my full hybrid pipeline, which worried me at first — I thought something was broken. I wrote a small debug script to print out the actual retrieved chunks side by side, and found the pipeline WAS working correctly (it retrieved different chunks, in different order), but my metric (precision/recall at top-6) was too blunt to notice, since it only checks if the right chunk is somewhere in the top 6, not where. Once I expanded my test set from 10 to 25 questions, the difference actually became visible: hybrid retrieval beat plain vector search by about 12% precision and 19% recall.

I also learned the hard way that free hosting tiers matter a lot for anything using `torch`. My first deployment on Render kept crashing because their free tier only gives 512MB RAM, which isn't enough for the embedding model + reranker together. I moved to Hugging Face Spaces instead, which gives 16GB free.

## Tech stack

- FastAPI for the backend
- tree-sitter for parsing code into functions/classes
- sentence-transformers for embeddings, plus a cross-encoder for reranking
- Qdrant (cloud, free tier) for the vector database
- networkx for the call graph
- rank-bm25 for keyword search
- Ollama (local) for generating the final answer — this only works when running locally, since the free hosting tier doesn't have Ollama installed. The deployed version shows the retrieved code directly instead.

## Running it locally

```
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on mac/linux
pip install -r requirements.txt
```

Add a `.env` file with:
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_key
```

Then:
```
uvicorn app.api.main:app --reload
```

To ingest a repo and ask a question:
```
curl -X POST localhost:8000/ingest -d '{"repo_url": "https://github.com/psf/requests"}'
curl -X POST localhost:8000/ask -d '{"repo": "psf__requests", "question": "how does it handle retries?"}'
```

## Eval results

Ran on 25 hand-labeled questions about the `requests` library:

| Method | Precision@6 | Recall@6 |
|---|---|---|
| Vector search only | 0.113 | 0.64 |
| + reranker | 0.120 | 0.72 |
| + graph + BM25 (full pipeline) | 0.127 | 0.76 |

## What I'd improve next

- The call graph currently uses simple regex matching to figure out which function calls which — it works but isn't always accurate. Doing this properly with the AST would be better.
- Only supports Python right now (tree-sitter has grammars for other languages too).
- No persistence between server restarts yet — has to re-index every time.
- Would like to fine-tune the reranker on my own eval questions once I have more of them.

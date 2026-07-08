"""Turn retrieved chunks into a grounded answer with file:line citations.

Uses a local Ollama model (free, no API key, runs entirely on your machine)
instead of a paid API. Falls back to a short summary (not a raw code dump)
when Ollama isn't reachable, e.g. in cloud deployments where it isn't
installed — the citation panel already shows the actual code, so the
fallback text doesn't need to repeat it.
"""

import json
import urllib.request

from app.retrieval.hybrid import RetrievalResult

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a precise codebase assistant. Answer the user's \
question using ONLY the provided code context. For every claim, cite the \
source using the exact citation string given (e.g. [app/utils.py#L12-L20]). \
If the context doesn't contain enough information to answer confidently, \
say so explicitly instead of guessing."""


def build_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"### {r.chunk.citation} ({r.chunk.node_type} {r.chunk.name})\n"
            f"```python\n{r.chunk.code}\n```"
        )
    return "\n\n".join(blocks)


def _short_fallback_summary(results: list[RetrievalResult]) -> str:
    """A brief, readable summary used when no LLM is available to generate
    a real answer — lists what was found without dumping full source code,
    since the frontend's citation panel already shows the code separately.
    """
    lines = [
        f"Generation isn't available in this environment (Ollama runs locally only). "
        f"Found {len(results)} relevant code locations — click a citation below to view each one:\n"
    ]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. `{r.chunk.citation}` — {r.chunk.node_type} `{r.chunk.name}`")
    return "\n".join(lines)


def answer_question(question: str, results: list[RetrievalResult]) -> dict:
    context = build_context(results)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n\n{context}\n\nQuestion: {question}"

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        answer = data.get("response", "")
    except Exception:
        answer = _short_fallback_summary(results)

    return {
        "answer": answer,
        "citations": [r.chunk.citation for r in results],
    }

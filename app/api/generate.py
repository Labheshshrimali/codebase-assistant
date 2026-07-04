"""Generation step: takes retrieved code chunks + a question, and produces
a cited answer using a local Ollama model instead of a paid API."""
import os
import requests as http

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
MAX_CHUNKS_IN_PROMPT = 3
MAX_CHARS_PER_CHUNK = 1200


def _build_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for c in chunks[:MAX_CHUNKS_IN_PROMPT]:
        code = c.chunk.code
        if len(code) > MAX_CHARS_PER_CHUNK:
            code = code[:MAX_CHARS_PER_CHUNK] + "\n... (truncated)"
        context_blocks.append(
            f"### {c.chunk.repo_relative_path} (lines {c.chunk.start_line}-{c.chunk.end_line})\n"
            f"```python\n{code}\n```"
        )
    context = "\n\n".join(context_blocks)
    return f"""You are a code assistant. Answer the question using ONLY the code below.
Cite the file and line numbers you used in your answer. Be concise.

{context}

Question: {question}

Answer (include file:line citations):"""


def generate_answer(question: str, retrieved_chunks: list) -> dict:
    prompt = _build_prompt(question, retrieved_chunks)
    try:
        resp = http.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        answer_text = data.get("response", "").strip()
    except Exception as e:
        answer_text = f"[generation failed: {e}]"

    citations = [
        {"file": c.chunk.repo_relative_path, "start_line": c.chunk.start_line, "end_line": c.chunk.end_line}
        for c in retrieved_chunks
    ]
    return {"answer": answer_text, "citations": citations}


answer_question = generate_answer
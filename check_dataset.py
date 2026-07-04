import json
import sys

MAX_REASONABLE_LINES = 40

def parse_chunk_id(chunk_id):
    path, line_range = chunk_id.rsplit(":", 1)
    start, end = line_range.split("-")
    return path, int(start), int(end)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "app/eval/qa_dataset.jsonl"
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    print(f"Checked {len(lines)} questions.\n")
    flagged = 0
    for i, item in enumerate(lines, 1):
        question = item.get("question", "<no question>")
        for cid in item.get("relevant_chunk_ids", []):
            try:
                _, start, end = parse_chunk_id(cid)
            except ValueError:
                print(f"[{i}] SKIP - can't parse chunk id: {cid}")
                continue
            span = end - start
            if span > MAX_REASONABLE_LINES:
                flagged += 1
                print(f"[{i}] TOO BROAD ({span} lines): {cid}")
                print(f"      Q: {question}")

    print(f"\n{flagged} chunk range(s) flagged out of total entries.")
    if flagged:
        print("Narrow these to the specific function/method that actually")
        print("answers the question, not the whole class/file it lives in.")

if __name__ == "__main__":
    main()

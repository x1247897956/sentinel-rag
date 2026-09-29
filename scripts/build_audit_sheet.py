"""Build a reproducible manual review sheet from the current answer snapshot."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from src.config import EVAL_DIR, REPORT_DIR
from src.retrieval import storage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--snapshot", type=Path, default=EVAL_DIR / "snapshot/answers.jsonl")
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "human_audit_current.csv")
    args = parser.parse_args()
    answers = [json.loads(line) for line in args.snapshot.read_text(encoding="utf-8").splitlines() if line]
    eligible = [a for a in answers if a.get("answerable") and not a.get("refused")]
    if not eligible:
        raise SystemExit("no non-refused answerable rows to audit")
    count = max(1, math.ceil(len(eligible) * args.rate))
    selected = sorted(random.Random(args.seed).sample(eligible, count), key=lambda a: a["qid"])
    chunk_ids = sorted({cid for answer in selected for cid in answer.get("allowed_ids", [])})
    conn = storage.connect()
    try:
        with conn.cursor() as cur:
            chunks = storage.fetch_chunks(cur, chunk_ids)
    finally:
        conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["qid", "question", "gold_points", "answer", "citations", "context_chunks", "human_covered", "human_note"]
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for answer in selected:
            context = [
                {"chunk_id": cid, "text": chunks.get(cid, {}).get("text", "MISSING")}
                for cid in answer.get("allowed_ids", [])
            ]
            writer.writerow({
                "qid": answer["qid"],
                "question": answer["question"],
                "gold_points": json.dumps(answer.get("gold_points", []), ensure_ascii=False),
                "answer": answer.get("answer", ""),
                "citations": json.dumps(answer.get("citations", []), ensure_ascii=False),
                "context_chunks": json.dumps(context, ensure_ascii=False),
                "human_covered": "",  # Human fills a true/false value per gold point.
                "human_note": "",
            })
    print(f"audit_sheet={args.out} sampled={count}/{len(eligible)} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

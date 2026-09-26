"""分块阶段：JSONL 语料 → 内存中的 Chunk 列表。

单独一层是为了让「抓取」（慢、依赖网络）与「分块入库」可分开重跑，
也方便在 CI 里用固定的语料快照复现评测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CHUNK_OVERLAP, CHUNK_TARGET, CORPUS_DIR
from src.ingest.chunker import Chunk, chunk_document
from src.ingest.schema import Doc


def load_corpus(paths: list[Path] | None = None) -> list[Doc]:
    paths = paths or sorted(p for p in CORPUS_DIR.glob("*.jsonl"))
    docs: list[Doc] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                docs.append(
                    Doc(
                        doc_id=rec["doc_id"],
                        source=rec["source"],
                        source_type=rec["source_type"],
                        title=rec["title"],
                        text=rec["text"],
                        published_at=rec.get("published_at"),
                        updated_at=rec.get("updated_at"),
                        cve_id=rec.get("cve_id"),
                        severity=rec.get("severity"),
                        cvss=rec.get("cvss"),
                        affected_versions=rec.get("affected_versions"),
                        meta=rec.get("meta") or {},
                    )
                )
    return docs


def build_chunks(
    docs: list[Doc] | None = None,
    target: int = CHUNK_TARGET,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[Doc], list[Chunk]]:
    docs = docs if docs is not None else load_corpus()
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, target=target, overlap=overlap))
    return docs, chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {"chunks": 0}
    lens = [len(c.text) for c in chunks]
    token_est = [len(c.tokens.split()) for c in chunks]
    lens_sorted = sorted(lens)
    return {
        "chunks": len(chunks),
        "avg_chars": round(sum(lens) / len(lens), 1),
        "p50_chars": lens_sorted[len(lens_sorted) // 2],
        "p95_chars": lens_sorted[int(len(lens_sorted) * 0.95) - 1],
        "max_chars": lens_sorted[-1],
        "avg_token_count": round(sum(token_est) / len(token_est), 1),
    }

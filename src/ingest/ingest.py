"""入库：分块 → 批量编码 embedding → 写 PostgreSQL（去重 + 增量 + stale）。

用法：
  uv run python -m src.ingest.ingest                 # 增量入库
  uv run python -m src.ingest.ingest --all           # 全量（先清空）
  uv run python -m src.ingest.ingest --limit 50      # 冒烟测试
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import CORPUS_DIR, EMBED_MODEL, REPO_ROOT_STR  # noqa: F401  (REPO_ROOT_STR 供日志)
from src.ingest.pipeline import build_chunks, chunk_stats, load_corpus
from src.retrieval import storage
from src.retrieval.models import Embedder


def ingest(all_docs: bool = False, limit: int | None = None, batch: int = 32, dump: Path | None = None) -> dict:
    t0 = time.time()
    docs = load_corpus()
    if limit:
        docs = docs[:limit]
    print(f"[ingest] 语料 {len(docs)} 篇")

    conn = storage.connect()
    # 先按 content_hash 判重：内容没变的文档**根本不进编码阶段**——
    # 否则一次「没有变化」的增量入库仍要付全量 embedding 的钱（实测 2619 chunk 约 350s）。
    if all_docs:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE runs, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()
    pending: list = []
    skipped_unchanged = 0
    with conn.cursor() as cur:
        for d in docs:
            cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (d.doc_id,))
            row = cur.fetchone()
            if row is not None and row[0] == d.to_record()["content_hash"]:
                skipped_unchanged += 1
                continue
            pending.append(d)
    if skipped_unchanged:
        print(f"[ingest] 内容未变、跳过编码的文档：{skipped_unchanged} 篇")

    docs, chunks = build_chunks(pending)
    stats = chunk_stats(chunks)
    print(f"[ingest] 待处理语料 {len(docs)} 篇，分块 {stats}")

    embed_s = 0.0
    vectors: list[list[float]] = []
    if chunks:
        print(f"[ingest] 开始编码 embedding（{EMBED_MODEL}）")
        embedder = Embedder()
        t_embed = time.time()
        vectors = embedder.encode(
            [f"{c.section} {c.text}" if c.section else c.text for c in chunks], batch_size=batch
        )
        embed_s = time.time() - t_embed
        print(f"[ingest] embedding 完成，{embed_s:.1f}s（{len(chunks) / max(embed_s, 1e-6):.0f} chunk/s）")
    counters = {
        "inserted_docs": 0,
        "unchanged_docs": skipped_unchanged,
        "updated_docs": 0,
        "inserted_chunks": 0,
        "skipped_chunks": 0,
    }
    try:
        with conn.cursor() as cur:
            by_doc: dict[str, list[int]] = {}
            for i, c in enumerate(chunks):
                by_doc.setdefault(c.doc_id, []).append(i)
            for n, doc in enumerate(docs, 1):
                rec = doc.to_record()
                action = storage.upsert_document(cur, rec)
                counters[f"{action}_docs" if action != "unchanged" else "unchanged_docs"] += 1
                if action == "unchanged":
                    counters["skipped_chunks"] += len(by_doc.get(doc.doc_id, []))
                    continue
                idxs = by_doc.get(doc.doc_id, [])
                n_ins = storage.insert_chunks(cur, rec, [chunks[i] for i in idxs], [vectors[i] for i in idxs])
                counters["inserted_chunks"] += n_ins
                counters["skipped_chunks"] += len(idxs) - n_ins
                if n % 100 == 0:
                    conn.commit()
                    print(f"[ingest] {n}/{len(docs)} 篇，已入库 chunk {counters['inserted_chunks']}")
            conn.commit()
            with conn.cursor() as c2:
                db_stats = storage.stats(c2)
            if dump:
                n = storage.dump_chunks_jsonl(c2 if False else cur, dump)
                print(f"[ingest] chunk 快照写出 {n} 行 → {dump}")
    finally:
        conn.close()

    result = {
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_stats": stats,
        "embed_seconds": round(embed_s, 1),
        "embed_model": EMBED_MODEL,
        "counters": counters,
        "db": db_stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "db"}, ensure_ascii=False, indent=2))
    print("[ingest] 库内状态：" + json.dumps(db_stats, ensure_ascii=False))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="清空后全量重建")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dump", type=Path, default=None, help="把入库后的 chunk 快照写成 JSONL")
    args = ap.parse_args()
    ingest(all_docs=args.all, limit=args.limit, batch=args.batch, dump=args.dump)
